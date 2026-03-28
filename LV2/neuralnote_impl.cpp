/**
 * NeuralNote Guitar2MIDI — LV2 Implementation (low-latency streaming)
 *
 * Compiled multiple times with different -march flags by CMake.
 * Loaded by neuralnote_guitar2midi.so (the wrapper) via dlopen().
 *
 * Threading model
 * ───────────────
 * One worker thread per note range.  All inter-thread communication is lockless:
 *
 *   run() → worker :  SnapshotChannel  (SPSC, atomic ready flag + POSIX semaphore)
 *   worker → run() :  MidiOutQueue     (SPSC ring buffer, atomic head/tail)
 *
 * run() writes into the ring buffer and checks the atomic ready flag.  When
 * MIN_FRESH_SAMPLES (25 ms) of new audio have arrived and the worker has consumed
 * the previous snapshot, it copies the ring into the snapshot slot, sets ready,
 * and posts the semaphore.  The worker sleeps on the semaphore, runs inference,
 * then pushes MIDI events to its per-range MidiOutQueue.  run() drains all
 * MidiOutQueues each cycle — no mutex anywhere on the hot path.
 *
 * Per-range configuration
 * ───────────────────────
 * If neuralnote_ranges.conf exists in the bundle directory each [range] section
 * defines a separate MIDI range.  Without it, single-range mode uses LV2 ports.
 */

#include <lv2/core/lv2.h>
#include <lv2/atom/atom.h>
#include <lv2/atom/forge.h>
#include <lv2/atom/util.h>
#include <lv2/midi/midi.h>
#include <lv2/urid/urid.h>
#include <lv2/log/log.h>
#include <lv2/log/logger.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <map>
#include <memory>
#include <semaphore.h>
#include <set>
#include <thread>
#include <vector>

// McLeod Pitch Method: only on ARMv8.2-A (Pi 5, Cortex-A76).
// Must be defined before NeuralNoteShared.h so that RangeStateBase and all
// shared pipeline functions compile in the McLeod call-sites.
// The neon (Pi 4) build leaves this undefined — no FFTW dependency.
#if defined(__aarch64__) && defined(__ARM_FEATURE_DOTPROD)
#  define NEURALNOTE_ENABLE_MPM 1
#endif

#include "BasicPitchConstants.h"
#include "NeuralNoteShared.h"  // pulls in BinaryData.h, BasicPitch.h, NoteRangeConfig.h,
                               // OneBitPitchDetector.h, McLeodPitchDetector.h (if MPM)
#include "SwiftF0Detector.h"

#define PLUGIN_URI "https://github.com/DamRsn/NeuralNote/guitar2midi"

#ifndef NEURALNOTE_IMPL_NAME
#define NEURALNOTE_IMPL_NAME "neuralnote_impl"
#endif

#define NEURALNOTE_STRINGIFY2(x) #x
#define NEURALNOTE_STRINGIFY(x)  NEURALNOTE_STRINGIFY2(x)

// ── Port indices ──────────────────────────────────────────────────────────────

enum PortIndex {
    PORT_AUDIO_IN        = 0,
    PORT_MIDI_OUT        = 1,
    PORT_AUDIO_OUT       = 2,
    PORT_THRESHOLD       = 3,
    PORT_GATE            = 4,
    PORT_AMP_FLOOR       = 5,
    PORT_FRAME_THRESHOLD = 6,
    PORT_MODE            = 7,
    PORT_ONSET_BLANK_MS  = 8,
};

// ── Mapped URIDs ──────────────────────────────────────────────────────────────

struct URIs {
    LV2_URID atom_Sequence;
    LV2_URID atom_EventTransfer;
    LV2_URID midi_MidiEvent;
};

static void mapURIs(LV2_URID_Map* map, URIs* uris)
{
    uris->atom_Sequence      = map->map(map->handle, LV2_ATOM__Sequence);
    uris->atom_EventTransfer = map->map(map->handle, LV2_ATOM__eventTransfer);
    uris->midi_MidiEvent     = map->map(map->handle, LV2_MIDI__MidiEvent);
}

// Forward declaration
struct NeuralNotePlugin;

// ── Per-range runtime state ───────────────────────────────────────────────────
// All common fields (including MidiOutQueue midiOut) live in RangeStateBase.

struct RangeState : RangeStateBase {
};

// ── Plugin instance ───────────────────────────────────────────────────────────

struct NeuralNotePlugin {
    LV2_URID_Map*   map;
    LV2_Log_Logger  logger;
    LV2_Atom_Forge  forge;
    URIs            uris;

    const float*       audioIn;
    float*             audioOut;
    LV2_Atom_Sequence* midiOut;
    const float*       thresholdPort;
    const float*       gate;
    const float*       ampFloor;
    const float*       frameThresholdPort;
    const float*       modePort;
    const float*       onsetBlankMsPort;

    double sampleRate;

    std::atomic<float> thresholdVal{0.6f};
    std::atomic<float> frameThresholdVal{0.5f};
    std::atomic<float> ampFloorVal{0.65f};
    std::atomic<int>   modeVal{1};  // 0 = poly, 1 = mono, 2 = swiftmono (default: mono)

    std::unique_ptr<SwiftF0Detector> swiftF0;  // null if model not found
    std::vector<float>               sf0Buf;   // worker-thread scratch: 16 kHz resampled audio

    std::vector<std::unique_ptr<RangeState>> ranges;
    bool singleRangeMode = true;

    // Onset detector state (run() thread only — no synchronisation needed)
    float onsetSmoothedRms = 0.001f;
    int   onsetBlankRemain = 0;

    // HPF pick detector (run() thread only)
    PickDetector pickDetector;

    // Onset-recency tracking (audio → worker, for decay-tail ghost suppression)
    std::atomic<uint64_t> totalSamples{0};
    std::atomic<uint64_t> lastOnsetSample{0};

    // Single worker thread shared across all ranges
    std::thread       workerThread;
    std::atomic<bool> workerQuit{false};
    sem_t             workerSem;
};

// ── Worker thread ──────────────────────────────────────────────────────────────

static void runWorker(NeuralNotePlugin* self)
{
    std::array<bool, 8> hasPriorResult = {};

    while (true) {
        sem_wait(&self->workerSem);
        if (self->workerQuit.load(std::memory_order_acquire)) break;

        const float ampFloor  = self->ampFloorVal.load(std::memory_order_relaxed);
        const int   modeNow   = self->modeVal.load(std::memory_order_relaxed);
        const bool  swiftMono = (modeNow == 2 && self->swiftF0 != nullptr);
        const bool  mono      = (modeNow >= 1);

        for (int ri = 0; ri < static_cast<int>(self->ranges.size()); ++ri) {
            RangeState& r = *self->ranges[ri];
            const bool hasSnap    = r.snapChan.ready.load(std::memory_order_acquire);

            {
                const float os       = self->thresholdVal.load(std::memory_order_relaxed);
                const float ft       = self->frameThresholdVal.load(std::memory_order_relaxed);
                const float minDurMs = r.cfg.minNoteLength * FFT_HOP / AUDIO_SAMPLE_RATE * 1000.0f;
                r.basicPitch->setParameters(1.0f - ft, os, minDurMs);
            }

            if (hasSnap) {
                // Two-phase: insert provisional before diff.
                // Staleness check: if a new provisional has fired since this snapshot
                // was dispatched, the CNN result doesn't correspond to the current
                // provisional — don't cancel it.
                const int prov = r.snapChan.provNoteAtDispatch;
                r.snapChan.provNoteAtDispatch = -1;

                int provForDiff = -1;
                if (prov != -1) {
                    const int currentProv = r.provNote.load(std::memory_order_acquire);
                    if (currentProv == prov) {
                        // CNN result matches current provisional — normal two-phase path
                        r.provNote.store(-1, std::memory_order_release);
                        if (prov >= NOTE_BASE && prov < NOTE_BASE + NOTE_COUNT) {
                            r.activeNotes |= (1ULL << (prov - NOTE_BASE));
                            provForDiff = prov;
                        } else {
                            r.midiOut.push({false, prov, 0});
                        }
                    }
                    // If stale (currentProv != prov): a new provisional has fired.
                    // Leave provNote intact; pass provForDiff = -1 to avoid cancelling it.
                }

                const uint64_t prevActive = r.activeNotes;
                uint64_t newBits = 0;
                int8_t   newVel[NOTE_COUNT] = {};

                if (swiftMono) {
                    // SwiftF0 path: resample 22050→16000 Hz, run ONNX inference.
                    const float* src    = r.snapChan.data.data();
                    const int    srcLen = r.snapChan.snapshotSize;
                    const bool   wasOnset = r.snapChan.onsetDispatched;
                    constexpr double SF0_SR = 16000.0;
                    const double ratio  = SF0_SR / PLUGIN_SR;
                    const int    dstLen = static_cast<int>(srcLen * ratio);
                    self->sf0Buf.resize(dstLen);
                    for (int i = 0; i < dstLen; ++i) {
                        const double pos = i / ratio;
                        const int    s0  = static_cast<int>(pos);
                        const double f   = pos - s0;
                        const int    s1  = std::min(s0 + 1, srcLen - 1);
                        self->sf0Buf[i]  = static_cast<float>((1.0 - f) * src[s0] + f * src[s1]);
                    }
                    r.snapChan.ready.store(false, std::memory_order_release);

                    const int midiNote = self->swiftF0->infer(
                        self->sf0Buf.data(), dstLen, 0.5f);

                    if (midiNote >= r.cfg.midiLow && midiNote <= r.cfg.midiHigh &&
                        midiNote >= NOTE_BASE && midiNote < NOTE_BASE + NOTE_COUNT) {
                        const int bit = midiNote - NOTE_BASE;
                        newBits    = 1ULL << bit;
                        newVel[bit] = 100;
                    }

                    // Onset-transition grace: the first snapshot(s) after an onset
                    // contain stale ring audio.  Two cases:
                    //   Mid-note: keep current note for 1 cycle (half-step suppression).
                    //   From silence: suppress cycle 1 and remember the stale note;
                    //     cycle 2 allows immediately if a DIFFERENT note is detected
                    //     (it's the real new note), or suppresses if same stale note.
                    if (wasOnset && r.swiftOnsetGrace <= 0) {
                        r.swiftOnsetGrace = 2;
                        r.swiftGraceStaleNote = -1;
                    }
                    if (r.swiftOnsetGrace > 0) {
                        if (r.activeNotes && newBits != r.activeNotes) {
                            // Mid-note transition: keep current note
                            newBits = r.activeNotes;
                            for (uint64_t tmp = newBits; tmp; tmp &= tmp - 1)
                                newVel[__builtin_ctzll(tmp)] = 100;
                        } else if (!r.activeNotes && newBits) {
                            const int detected = NOTE_BASE + __builtin_ctzll(newBits);
                            if (r.swiftGraceStaleNote < 0) {
                                // Cycle 1: suppress and remember the stale note
                                r.swiftGraceStaleNote = detected;
                                newBits = 0;
                            } else if (detected == r.swiftGraceStaleNote) {
                                // Cycle 2: same stale note → still suppress
                                newBits = 0;
                            }
                            // else: different note on cycle 2 → allow (real note)
                        }
                        --r.swiftOnsetGrace;
                    }

                    // Decay-tail ghost suppression: if SwiftF0 wants to create a
                    // new note-ON but no onset fired recently, this is likely a
                    // harmonic from a decaying string.  In mono/swiftmono every
                    // legitimate new note starts with an onset.
                    if (newBits & ~r.activeNotes) {
                        const uint64_t elapsed =
                            self->totalSamples.load(std::memory_order_relaxed)
                            - self->lastOnsetSample.load(std::memory_order_acquire);
                        constexpr double ONSET_GATE_S = 0.25;
                        const auto gate = static_cast<uint64_t>(self->sampleRate * ONSET_GATE_S);
                        if (elapsed > gate)
                            newBits = 0;
                    }
                } else {
                    r.basicPitch->transcribeToMIDI(r.snapChan.data.data(),
                                                    r.snapChan.snapshotSize);
                    r.snapChan.ready.store(false, std::memory_order_release);
                    hasPriorResult[ri] = true;
                    buildNNBits(r, ampFloor, newBits, newVel);
                }

                // Cancel grace: first CNN cycle after a new provisional fires often
                // contains mostly the previous note's audio.  Suppress that cancel so
                // the next cycle (with a full window of the new note) decides instead.
                // Skipped in swiftmono: SwiftF0 is fast enough to correct wrong
                // provisionals immediately; onset grace handles mixed-audio transitions.
                if (!swiftMono) {
                    if (provForDiff != -1 && provForDiff != r.provLastSeenByCNN) {
                        r.provLastSeenByCNN = provForDiff;
                        r.provCancelGrace   = 1;
                    }
                    if (provForDiff != -1 && r.provCancelGrace > 0) {
                        const int bit = provForDiff - NOTE_BASE;
                        if (bit >= 0 && bit < NOTE_COUNT && !(newBits & (1ULL << bit))) {
                            --r.provCancelGrace;
                            newBits    |= (1ULL << bit);  // suppress cancel; note already playing
                            newVel[bit] = 64;             // velocity unused (no new note-ON fired)
                        } else {
                            r.provCancelGrace = 0;  // CNN confirmed naturally
                        }
                    }
                }

                applyNotesDiff(r, newBits, newVel, provForDiff, mono,
                               swiftMono ? r.cfg.swiftHoldCycles : -1);

                // Mono cross-range: new note-ON(s) in this range → kill all other ranges
                if (mono && (r.activeNotes & ~prevActive)) {
                    for (int oi = 0; oi < static_cast<int>(self->ranges.size()); ++oi) {
                        if (oi == ri) continue;
                        RangeState& other = *self->ranges[oi];
                        for (uint64_t tmp = other.activeNotes; tmp; tmp &= tmp - 1)
                            other.midiOut.push({false,
                                NOTE_BASE + static_cast<int>(__builtin_ctzll(tmp)), 0});
                        other.activeNotes = 0;
                        other.holdNotes   = 0;
                        std::memset(other.holdCounts, 0, NOTE_COUNT);
                        other.monoHeldNote.store(-1, std::memory_order_release);
                    }
                }
            }
        }
    }

    // Shutdown: note-offs for all active notes across all ranges
    for (auto& rp : self->ranges) {
        for (uint64_t tmp = rp->activeNotes; tmp; tmp &= tmp - 1)
            rp->midiOut.push({false, NOTE_BASE + static_cast<int>(__builtin_ctzll(tmp)), 0});
        rp->activeNotes = 0;
        rp->holdNotes   = 0;
    }
}

// ── Helpers ───────────────────────────────────────────────────────────────────

static void writeMidi(LV2_Atom_Forge* forge, uint32_t frames,
                       LV2_URID midiType, uint8_t b0, uint8_t b1, uint8_t b2)
{
    uint8_t msg[3] = {b0, b1, b2};
    lv2_atom_forge_frame_time(forge, frames);
    lv2_atom_forge_atom(forge, 3, midiType);
    lv2_atom_forge_write(forge, msg, 3);
}

static void pushToRing(RangeState& r, double sampleRate,
                        const float* in, int inLen)
{
    const int rs = r.ringSize;
    if (sampleRate == 22050.0) {
        for (int i = 0; i < inLen; ++i) {
            r.ring[r.ringHead] = in[i];
            r.ringHead = (r.ringHead + 1) % rs;
            if (r.ringFilled < rs) ++r.ringFilled;
        }
        r.freshSamples += inLen;
        return;
    }
    const double ratio  = 22050.0 / sampleRate;
    const int    outLen = static_cast<int>(inLen * ratio);
    for (int i = 0; i < outLen; ++i) {
        const double srcPos = i / ratio;
        const int    s0     = static_cast<int>(srcPos);
        const double frac   = srcPos - s0;
        const int    s1     = std::min(s0 + 1, inLen - 1);
        r.ring[r.ringHead] =
            static_cast<float>((1.0 - frac) * in[s0] + frac * in[s1]);
        r.ringHead = (r.ringHead + 1) % rs;
        if (r.ringFilled < rs) ++r.ringFilled;
    }
    r.freshSamples += outLen;
}

// ── Lifecycle ─────────────────────────────────────────────────────────────────

static LV2_Handle instantiate(const LV2_Descriptor*,
                                double                rate,
                                const char*           bundlePath,
                                const LV2_Feature* const* features)
{
    NeuralNotePlugin* self = new NeuralNotePlugin();
    self->sampleRate = rate;
    self->map        = nullptr;

    for (int i = 0; features[i]; ++i) {
        if (!strcmp(features[i]->URI, LV2_URID__map))
            self->map = static_cast<LV2_URID_Map*>(features[i]->data);
        else if (!strcmp(features[i]->URI, LV2_LOG__log))
            lv2_log_logger_init(&self->logger, self->map,
                                static_cast<LV2_Log_Log*>(features[i]->data));
    }
    if (!self->map) { delete self; return nullptr; }

    mapURIs(self->map, &self->uris);
    lv2_atom_forge_init(&self->forge, self->map);

    try { BinaryData::init(bundlePath); }
    catch (const std::exception& e) {
        lv2_log_error(&self->logger, "NeuralNote: %s\n", e.what());
        delete self;
        return nullptr;
    }

    // Try to load SwiftF0 model from bundle (optional — swiftmono mode degrades to
    // basicpitch if the file is absent).
    {
        std::string sf0Path = std::string(bundlePath);
        if (!sf0Path.empty() && sf0Path.back() != '/') sf0Path += '/';
        sf0Path += "swift_f0_model.onnx";
        try {
            self->swiftF0 = std::make_unique<SwiftF0Detector>(sf0Path);
            lv2_log_note(&self->logger, "NeuralNote: SwiftF0 model loaded from %s\n",
                         sf0Path.c_str());
        } catch (const std::exception& e) {
            lv2_log_note(&self->logger,
                         "NeuralNote: SwiftF0 model not loaded (%s) — swiftmono will use BasicPitch\n",
                         e.what());
        }
    }

    std::string cfgPath = std::string(bundlePath);
    if (!cfgPath.empty() && cfgPath.back() != '/') cfgPath += '/';
    cfgPath += "neuralnote_ranges.conf";
    RangeConfig rangeCfg = loadRangeConfig(cfgPath);

    auto makeRange = [&](const NoteRange& cfg) {
        auto r             = std::make_unique<RangeState>();
        r->cfg             = cfg;
        r->ringSize        = windowMsToRingSize(cfg.windowMs);
        r->minFreshSamples = std::max(r->ringSize / 2, MIN_FRESH_FLOOR);
        r->ring.assign(RING_MAX, 0.0f);
        r->basicPitch = std::make_unique<BasicPitch>();
        const float minDurMs = cfg.minNoteLength * FFT_HOP / AUDIO_SAMPLE_RATE * 1000.0f;
        r->basicPitch->setParameters(1.0f - self->frameThresholdVal.load(std::memory_order_relaxed),
                                     self->thresholdVal.load(std::memory_order_relaxed), minDurMs);
        // Configure per-range OBP lowpass
        const float sr     = static_cast<float>(rate);
        const float cutoff = std::min(440.0f * std::pow(2.0f, (cfg.midiHigh - 69) / 12.0f) * 1.2f,
                                      sr * 0.45f);
        r->obd.setLowpass(cutoff, sr);
#ifdef NEURALNOTE_ENABLE_MPM
        r->mpm.init(sr, cfg.midiLow, cfg.midiHigh);
#endif
        return r;
    };

    if (!rangeCfg.ranges.empty()) {
        self->singleRangeMode = false;
        // Global params (threshold, frame_threshold, amp_floor, mode) come from
        // LV2 port defaults (plugin.ttl) — not from the conf file. The host
        // delivers them via connect_port + run() before any audio is processed.
        for (const auto& rc : rangeCfg.ranges)
            self->ranges.push_back(makeRange(rc));
        lv2_log_note(&self->logger, "NeuralNote: loaded %zu ranges from %s\n",
                     self->ranges.size(), cfgPath.c_str());
    } else {
        self->singleRangeMode = true;
        NoteRange def;
        def.midiLow = 0; def.midiHigh = 127; def.name = "default";
        def.windowMs = 150.0f; def.minNoteLength = 6; def.holdCycles = 2;
        self->ranges.push_back(makeRange(def));
    }

    self->pickDetector.init(static_cast<float>(rate));

    sem_init(&self->workerSem, 0, 0);
    self->workerThread = std::thread(runWorker, self);

    lv2_log_note(&self->logger,
                 "NeuralNote Guitar2MIDI: %.0f Hz  [impl: " NEURALNOTE_IMPL_NAME
                 "]  [ranges: %zu, single worker thread]\n", rate, self->ranges.size());
    return static_cast<LV2_Handle>(self);
}

static void connectPort(LV2_Handle instance, uint32_t port, void* data)
{
    NeuralNotePlugin* self = static_cast<NeuralNotePlugin*>(instance);
    switch (static_cast<PortIndex>(port)) {
        case PORT_AUDIO_IN:        self->audioIn            = static_cast<const float*>(data);       break;
        case PORT_MIDI_OUT:        self->midiOut            = static_cast<LV2_Atom_Sequence*>(data); break;
        case PORT_AUDIO_OUT:       self->audioOut           = static_cast<float*>(data);             break;
        case PORT_THRESHOLD:       self->thresholdPort      = static_cast<const float*>(data);       break;
        case PORT_GATE:            self->gate               = static_cast<const float*>(data);       break;
        case PORT_AMP_FLOOR:       self->ampFloor           = static_cast<const float*>(data);       break;
        case PORT_FRAME_THRESHOLD: self->frameThresholdPort = static_cast<const float*>(data);       break;
        case PORT_MODE:            self->modePort           = static_cast<const float*>(data);       break;
        case PORT_ONSET_BLANK_MS:  self->onsetBlankMsPort   = static_cast<const float*>(data);       break;
    }
}

static void activate(LV2_Handle instance)
{
    NeuralNotePlugin* self = static_cast<NeuralNotePlugin*>(instance);
    for (auto& rp : self->ranges) {
        rp->ringHead     = 0;
        rp->ringFilled   = 0;
        rp->freshSamples = 0;
        std::fill(rp->ring.begin(), rp->ring.end(), 0.0f);
        rp->basicPitch->reset();
        rp->obd.reset();
        rp->obdVoting.reset();
#ifdef NEURALNOTE_ENABLE_MPM
        rp->mpm.reset();
#endif
        rp->obdOnsetActive     = false;
        rp->obdWindowRemain    = 0;
        rp->provNote.store(-1, std::memory_order_relaxed);
        rp->snapChan.provNoteAtDispatch = -1;
        // Send note-offs for any active notes
        for (uint64_t tmp = rp->activeNotes; tmp; tmp &= tmp - 1)
            rp->midiOut.push({false, NOTE_BASE + static_cast<int>(__builtin_ctzll(tmp)), 0});
        rp->activeNotes = 0;
        rp->holdNotes   = 0;
        std::memset(rp->holdCounts, 0, NOTE_COUNT);
        // Cancel any pending snapshot
        rp->snapChan.ready.store(false, std::memory_order_release);
    }
}

// ── run() ─────────────────────────────────────────────────────────────────────

static void run(LV2_Handle instance, uint32_t nSamples)
{
    NeuralNotePlugin* self = static_cast<NeuralNotePlugin*>(instance);

    lv2_atom_forge_set_buffer(&self->forge,
                               reinterpret_cast<uint8_t*>(self->midiOut),
                               self->midiOut->atom.size);
    LV2_Atom_Forge_Frame seqFrame;
    lv2_atom_forge_sequence_head(&self->forge, &seqFrame, 0);

    // Drain all per-range MIDI output queues — lockless
    for (auto& rp : self->ranges) {
        PendingNote pn;
        while (rp->midiOut.pop(pn))
            writeMidi(&self->forge, 0, self->uris.midi_MidiEvent,
                      pn.noteOn ? uint8_t(0x90) : uint8_t(0x80),
                      static_cast<uint8_t>(pn.pitch),
                      pn.noteOn ? static_cast<uint8_t>(pn.velocity) : uint8_t(0));
    }

    // Propagate LV2 port values to worker atomics every callback.
    // No change-detection: the worker must always see the host's current value,
    // including values restored from a preset or snapshot before the first run().
    if (self->thresholdPort)      self->thresholdVal.store(*self->thresholdPort, std::memory_order_relaxed);
    if (self->frameThresholdPort) self->frameThresholdVal.store(*self->frameThresholdPort, std::memory_order_relaxed);
    if (self->modePort)           self->modeVal.store(static_cast<int>(*self->modePort + 0.5f), std::memory_order_relaxed);
    if (self->ampFloor)           self->ampFloorVal.store(*self->ampFloor, std::memory_order_relaxed);

    if (self->audioOut)
        std::memset(self->audioOut, 0, nSamples * sizeof(float));

    // Noise gate + onset detection (share one RMS computation)
    const float gateFloor = (self->gate && *self->gate > 0.0f) ? *self->gate : 0.003f;
    float sumSq = 0.0f;
    for (uint32_t i = 0; i < nSamples; ++i)
        sumSq += self->audioIn[i] * self->audioIn[i];
    const float blockRms = std::sqrt(sumSq / static_cast<float>(nSamples));
    const bool  gated    = (blockRms < gateFloor);

    // Onset detection: HPF pick detector is primary, RMS is fallback.
    self->totalSamples.fetch_add(nSamples, std::memory_order_relaxed);
    bool onsetFired = false;

    // Primary: HPF pick detector with two-tier confirmation.
    //   Tier 1 (ratio ≥ 10): high confidence — immediate onset.
    //   Tier 2 (ratio 4–10): tentative — confirmed only if RMS also agrees.
    constexpr float PICK_HIGH_TIER = 10.0f;
    int   pickSample = -1;
    float pickRatio  = 0.0f;
    if (!gated)
        pickSample = self->pickDetector.process(
            self->audioIn, static_cast<int>(nSamples), pickRatio);

    const bool rmsWouldFire = !gated && self->onsetBlankRemain <= 0
                              && blockRms > self->onsetSmoothedRms * ONSET_RATIO;

    if (pickSample >= 0 && (pickRatio >= PICK_HIGH_TIER || rmsWouldFire)) {
        onsetFired = true;
        const float blankMs = (self->onsetBlankMsPort && *self->onsetBlankMsPort > 0.0f)
                                  ? *self->onsetBlankMsPort : ONSET_BLANK_MS;
        self->onsetBlankRemain = static_cast<int>(self->sampleRate * (blankMs / 1000.0f));
        self->onsetSmoothedRms = blockRms;
        self->lastOnsetSample.store(
            self->totalSamples.load(std::memory_order_relaxed)
                - static_cast<uint64_t>(nSamples) + static_cast<uint64_t>(pickSample),
            std::memory_order_release);
    }

    // Fallback: RMS onset (hammer-ons, volume swells, etc.)
    if (!onsetFired) {
        if (self->onsetBlankRemain > 0) {
            self->onsetBlankRemain -= static_cast<int>(nSamples);
            if (self->onsetBlankRemain < 0) self->onsetBlankRemain = 0;
        } else if (rmsWouldFire) {
            onsetFired             = true;
            const float blankMs    = (self->onsetBlankMsPort && *self->onsetBlankMsPort > 0.0f)
                                         ? *self->onsetBlankMsPort : ONSET_BLANK_MS;
            self->onsetBlankRemain = static_cast<int>(self->sampleRate * (blankMs / 1000.0f));
            self->onsetSmoothedRms = blockRms;
            self->lastOnsetSample.store(self->totalSamples.load(std::memory_order_relaxed),
                                        std::memory_order_release);
        }
    }
    if (!onsetFired && self->onsetBlankRemain == 0)
        self->onsetSmoothedRms = self->onsetSmoothedRms * (1.0f - ONSET_ALPHA) + blockRms * ONSET_ALPHA;

    static const float zeros[8192] = {};

    // Push audio into each range's ring; dispatch snapshot when ready — lockless
    for (auto& rp : self->ranges) {
        RangeState& r = *rp;

        if (!gated) {
            pushToRing(r, self->sampleRate, self->audioIn, static_cast<int>(nSamples));

            // Two-phase OneBitPitch + MPM: arm on onset, expire after 100 ms.
            armOrExpireOBP(r, static_cast<float>(self->sampleRate),
                           static_cast<int>(nSamples), onsetFired);

            // Helper: fire a provisional note — shared by immediate and pending paths.
            auto fireProv = [&](int note) {
                // Skip if the same note is already active — avoids redundant
                // re-triggers from RMS onset re-fires on the same note's energy.
                // Genuine staccato re-hits are detected by the pick detector
                // which resets the onset cycle; the worker's applyNotesDiff
                // handles the actual OFF+ON via the hold/re-hit path.
                if (note >= NOTE_BASE && note < NOTE_BASE + NOTE_COUNT) {
                    const uint64_t bits = r.activeNotesBits.load(std::memory_order_acquire);
                    if (bits & (1ULL << (note - NOTE_BASE)))
                        return;  // note already playing — skip redundant provisional
                }
                // Mono / SwiftMono: kill any other range's held note
                if (self->modeVal.load(std::memory_order_relaxed) >= 1) {
                    for (auto& other : self->ranges) {
                        const int held = other->monoHeldNote.load(std::memory_order_acquire);
                        if (held != -1 && held != note) {
                            writeMidi(&self->forge, 0, self->uris.midi_MidiEvent,
                                      0x80, static_cast<uint8_t>(held), uint8_t(0));
                            other->monoHeldNote.store(-1, std::memory_order_release);
                        }
                    }
                }
                writeMidi(&self->forge, 0, self->uris.midi_MidiEvent,
                          0x90, static_cast<uint8_t>(note), uint8_t(100));
                r.provNote.store(note, std::memory_order_release);
                r.monoHeldNote.store(note, std::memory_order_release);
            };

#ifdef NEURALNOTE_ENABLE_MPM
            // Push to MPM while OBP window is active OR while awaiting MPM on a
            // pending OBP vote — so we can retry analyze() next callback.
            if (r.obdOnsetActive || r.obdPendingNote != -1)
                r.mpm.push(self->audioIn, static_cast<int>(nSamples));
#endif

            if (r.obdOnsetActive) {
                if (r.provNote.load(std::memory_order_relaxed) == -1) {
                    const int finalNote = runOBPHPS(r, self->audioIn,
                                                    static_cast<int>(nSamples),
                                                    static_cast<float>(self->sampleRate),
                                                    self->ranges);
                    if (finalNote != -1) {
                        bool shouldFire = false;
#ifdef NEURALNOTE_ENABLE_MPM
                        const int mpmNote = r.mpm.analyze(
                            static_cast<float>(self->sampleRate),
                            r.cfg.midiLow, r.cfg.midiHigh);
                        if (mpmNote != -1) {
                            // OBP confirms onset; trust MPM for the pitch.
                            // Mono: suppress if any other range already has a provisional.
                            // Poly: suppress if MPM's note is a harmonic (±12/24 semitones).
                            const bool mono = (self->modeVal.load(std::memory_order_relaxed) >= 1);
                            bool suppress = false;
                            for (int oi = 0; oi < static_cast<int>(self->ranges.size()); ++oi) {
                                if (oi == ri) continue;
                                const int op2 = self->ranges[oi]->provNote.load(std::memory_order_relaxed);
                                if (op2 != -1) {
                                    if (mono) { suppress = true; break; }
                                    const int ad = std::abs(mpmNote - op2);
                                    if (ad == 12 || ad == 24) { suppress = true; break; }
                                }
                            }
                            if (!suppress) {
                                shouldFire = true;
                                finalNote  = mpmNote;
                            }
                        } else {
                            // MPM not ready — save vote and retry next callbacks
                            r.obdPendingNote   = finalNote;
                            r.obdPendingRemain = static_cast<int>(self->sampleRate * 0.1f);
                        }
#else
                        shouldFire = true;
#endif
                        if (shouldFire) fireProv(finalNote);
                    }
                }
            }
#ifdef NEURALNOTE_ENABLE_MPM
            else if (r.obdPendingNote != -1
                     && r.provNote.load(std::memory_order_relaxed) == -1) {
                // OBP voted previously but MPM wasn't ready — retry now that more
                // audio has accumulated.
                r.obdPendingRemain -= static_cast<int>(nSamples);
                if (r.obdPendingRemain <= 0) {
                    r.obdPendingNote = -1;  // timed out — give up
                } else {
                    const int mpmNote = r.mpm.analyze(
                        static_cast<float>(self->sampleRate),
                        r.cfg.midiLow, r.cfg.midiHigh);
                    if (mpmNote != -1) {
                        r.obdPendingNote = -1;
                        // Mono: suppress if any other range has a provisional.
                        // Poly: suppress harmonics (±12/24).
                        const bool mono = (self->modeVal.load(std::memory_order_relaxed) >= 1);
                        bool suppress = false;
                        for (int oi = 0; oi < static_cast<int>(self->ranges.size()); ++oi) {
                            if (&*self->ranges[oi] == &r) continue;
                            const int op2 = self->ranges[oi]->provNote.load(std::memory_order_relaxed);
                            if (op2 != -1) {
                                if (mono) { suppress = true; break; }
                                const int ad = std::abs(mpmNote - op2);
                                if (ad == 12 || ad == 24) { suppress = true; break; }
                            }
                        }
                        if (!suppress) fireProv(mpmNote);
                    }
                    // mpmNote == -1: still not ready, try again next callback
                }
            }
#endif
        } else {
            // Silence: clear OBP / MPM state so stale data can't bleed into next note
            resetOBPOnGate(r);
            int rem = static_cast<int>(nSamples);
            while (rem > 0) {
                const int chunk = std::min(rem, 8192);
                pushToRing(r, self->sampleRate, zeros, chunk);
                rem -= chunk;
            }
        }

        // Dispatch snapshot: linearise ring → snapshot slot, wake worker.
        dispatchSnapshotIfReady(r, onsetFired, 0.0, self->workerSem, gateFloor);
    }

    // Second drain: catch any CNN events the worker pushed during this callback.
    // The first drain (top of run) caught events from before this cycle;
    // this one reduces CNN note latency by up to one JACK buffer.
    for (auto& rp : self->ranges) {
        PendingNote pn;
        while (rp->midiOut.pop(pn))
            writeMidi(&self->forge, 0, self->uris.midi_MidiEvent,
                      pn.noteOn ? uint8_t(0x90) : uint8_t(0x80),
                      static_cast<uint8_t>(pn.pitch),
                      pn.noteOn ? static_cast<uint8_t>(pn.velocity) : uint8_t(0));
    }

    lv2_atom_forge_pop(&self->forge, &seqFrame);
}

static void deactivate(LV2_Handle /*instance*/) {}

static void cleanup(LV2_Handle instance)
{
    NeuralNotePlugin* self = static_cast<NeuralNotePlugin*>(instance);
    self->workerQuit.store(true, std::memory_order_release);
    sem_post(&self->workerSem);
    if (self->workerThread.joinable())
        self->workerThread.join();
    sem_destroy(&self->workerSem);
    delete self;
}

static const void* extensionData(const char* /*uri*/) { return nullptr; }

// ── Descriptor ───────────────────────────────────────────────────────────────

static const LV2_Descriptor descriptor = {
    PLUGIN_URI, instantiate, connectPort, activate,
    run, deactivate, cleanup, extensionData,
};

LV2_SYMBOL_EXPORT const LV2_Descriptor* neuralnote_impl_descriptor(uint32_t index)
{
    return (index == 0) ? &descriptor : nullptr;
}
