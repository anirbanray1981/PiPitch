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

// BinaryData.h must come before any Lib/Model header
#include "BinaryData.h"
#include "BasicPitch.h"
#include "BasicPitchConstants.h"
#include "NoteRangeConfig.h"
#include "OneBitPitchDetector.h"
#include "NeuralNoteShared.h"

// McLeod Pitch Method: only on ARMv8.2-A (Pi 5, Cortex-A76).
// __ARM_FEATURE_DOTPROD is defined when compiling with -march=armv8.2-a+dotprod.
// The neon (Pi 4) build omits MPM entirely — no FFTW dependency.
#if defined(__aarch64__) && defined(__ARM_FEATURE_DOTPROD)
#  define NEURALNOTE_ENABLE_MPM 1
#  include "McLeodPitchDetector.h"
#endif

#define PLUGIN_URI "https://github.com/DamRsn/NeuralNote/guitar2midi"

#ifndef NEURALNOTE_IMPL_NAME
#define NEURALNOTE_IMPL_NAME "neuralnote_impl"
#endif

#define NEURALNOTE_STRINGIFY2(x) #x
#define NEURALNOTE_STRINGIFY(x)  NEURALNOTE_STRINGIFY2(x)

// ── Port indices ──────────────────────────────────────────────────────────────

enum PortIndex {
    PORT_AUDIO_IN         = 0,
    PORT_MIDI_OUT         = 1,
    PORT_THRESHOLD        = 2,
    PORT_MODE             = 3,
    PORT_AUDIO_OUT        = 4,
    PORT_GATE             = 5,
    PORT_MIN_DUR          = 6,
    PORT_AMP_FLOOR        = 7,
    PORT_FRAME_THRESHOLD  = 8,
    PORT_MIN_NOTE_LENGTH  = 9,
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

// ── Pending MIDI event ────────────────────────────────────────────────────────

struct PendingNote { bool noteOn; uint8_t pitch; uint8_t velocity; };

// ── Lockless SPSC MIDI output queue (worker → run()) ─────────────────────────
// (SnapshotChannel is defined in NeuralNoteShared.h)
//
// Standard single-producer / single-consumer ring buffer.
// Producer: worker thread.  Consumer: run().

struct MidiOutQueue {
    PendingNote      buf[MIDI_QUEUE_CAP];
    std::atomic<int> head{0};   // run() advances
    std::atomic<int> tail{0};   // worker advances

    // Called from worker thread only.
    void push(PendingNote n) {
        const int t    = tail.load(std::memory_order_relaxed);
        const int next = (t + 1) % MIDI_QUEUE_CAP;
        if (next == head.load(std::memory_order_acquire)) return; // drop if full
        buf[t] = n;
        tail.store(next, std::memory_order_release);
    }

    // Called from run() only.
    bool pop(PendingNote& out) {
        const int h = head.load(std::memory_order_relaxed);
        if (h == tail.load(std::memory_order_acquire)) return false;
        out = buf[h];
        head.store((h + 1) % MIDI_QUEUE_CAP, std::memory_order_release);
        return true;
    }
};

// Forward declaration
struct NeuralNotePlugin;

// ── Per-range runtime state ───────────────────────────────────────────────────

struct PerRangeState {
    NoteRange cfg;

    std::unique_ptr<BasicPitch> basicPitch;

    // Ring buffer (22050-Hz resampled audio)
    std::vector<float> ringBuf;
    int ringHead         = 0;
    int ringFilled       = 0;
    int freshSamples     = 0;
    int ringSize         = 0;
    int minFreshSamples  = 0;  // = max(ringSize/2, MIN_FRESH_FLOOR); set at init/resize

    // Lockless inter-thread channels
    SnapshotChannel snapChan;   // run() → worker
    MidiOutQueue    midiOut;    // worker → run()

    // Note tracking (worker thread only — no synchronisation needed)
    // Note state — bus-aligned bitmaps over the guitar range E2–E6 (49 notes)
    uint64_t activeNotes = 0;
    uint64_t holdNotes   = 0;
    alignas(8) int8_t holdCounts[NOTE_COUNT] = {};

    // Two-phase pitch detection: OneBitPitch fires a provisional note-ON
    // immediately; BasicPitch CNN confirms, corrects, or cancels it.
    OneBitPitchDetector obd;
    std::atomic<int>    provNote{-1};  // -1 = none; set by run(), cleared by worker
    OBPVotingBuffer     obdVoting;
    std::atomic<int>    obdBlacklistNote{-1};  // note CNN cancelled; suppressed for next onset
    // Bit-parallel HPS register: accumulates all OBP detections this onset window.
    // Merged cross-range at voting time to identify the true fundamental.
    uint64_t            obpHpsBits = 0;
    bool                obdOnsetActive  = false;
    int                 obdWindowRemain = 0;

#ifdef NEURALNOTE_ENABLE_MPM
    // McLeod Pitch Method: runs in parallel with OBP during the onset window.
    // Provisional fires only if OBP + HPS + MPM all agree on the same note.
    McLeodPitchDetector mpm;
#endif

    // Single-range mode: run() writes these; worker reads them.
    // Ordering guaranteed by the semaphore post/wait on snapChan.
    std::atomic<float> onsetSensitivity{0.6f};
    std::atomic<float> frameThresholdVal{0.5f};
    std::atomic<float> minNoteFramesVal{6.0f};
    std::atomic<bool>  paramsChanged{false};

    // Non-copyable / non-moveable (owns SnapshotChannel which owns sem_t)
    PerRangeState()                              = default;
    PerRangeState(const PerRangeState&)          = delete;
    PerRangeState& operator=(const PerRangeState&) = delete;
    PerRangeState(PerRangeState&&)               = delete;
    PerRangeState& operator=(PerRangeState&&)    = delete;
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
    const float*       threshold;
    const float*       windowMs;
    const float*       gate;
    const float*       minDur;
    const float*       ampFloor;
    const float*       frameThresholdPort;
    const float*       minNoteLengthPort;

    double sampleRate;

    std::atomic<float> ampFloorVal{0.65f};

    // Change-detection (run() only — single-threaded)
    float lastThreshold{0.6f};
    float lastFrameThreshold{0.5f};
    float lastMinDur{30.0f};
    float lastMinNoteFrames{6.0f};
    float lastAmpFloor{0.65f};

    std::vector<std::unique_ptr<PerRangeState>> ranges;
    bool singleRangeMode = true;

    // Onset detector state (run() thread only — no synchronisation needed)
    float onsetSmoothedRms = 0.001f;
    int   onsetBlankRemain = 0;

    // Single worker thread shared across all ranges
    std::thread       workerThread;
    std::atomic<bool> workerQuit{false};
    sem_t             workerSem;
};

// ── Worker thread (one per range) ─────────────────────────────────────────────

static void applyRangeDiff(PerRangeState& r, float ampFloor, int prov = -1)
{
    // Build CNN result as bitmap + velocity array
    uint64_t newBits = 0;
    int8_t   newVel[NOTE_COUNT] = {};
    for (const auto& ev : r.basicPitch->getNoteEvents()) {
        if (static_cast<float>(ev.amplitude) < ampFloor) continue;
        const int p = static_cast<int>(ev.pitch);
        if (p < r.cfg.midiLow || p > r.cfg.midiHigh) continue;
        if (p < NOTE_BASE || p >= NOTE_BASE + NOTE_COUNT) continue;
        const int bit = p - NOTE_BASE;
        const int v   = std::clamp(static_cast<int>(ev.amplitude * 127.0), 1, 127);
        if (!(newBits & (1ULL << bit)) || v > newVel[bit]) {
            newBits      |= (1ULL << bit);
            newVel[bit]   = static_cast<int8_t>(v);
        }
    }

    // CNN did not confirm provisional — blacklist it for next onset (one-onset suppression).
    if (prov != -1 && !bmTest(newBits, prov))
        r.obdBlacklistNote.store(prov, std::memory_order_release);

    // If the provisional is already in the hold-countdown queue from a previous cycle,
    // force-expire it now so the decrement loop below sends OFF this cycle.
    if (prov != -1 && prov >= NOTE_BASE && prov < NOTE_BASE + NOTE_COUNT
        && !bmTest(newBits, prov)) {
        const int provBit = prov - NOTE_BASE;
        if (r.holdNotes & (1ULL << provBit))
            r.holdCounts[provBit] = 1;  // decrement to 0 below → OFF this cycle
    }

    // Note-ONs: cancel any hold for returning notes, then fire
    const uint64_t returning = newBits & r.holdNotes;
    for (uint64_t tmp = returning; tmp; tmp &= tmp - 1) {
        const int bit = __builtin_ctzll(tmp);
        r.holdCounts[bit] = 0;
    }
    r.holdNotes &= ~returning;
    for (uint64_t tmp = newBits & ~r.activeNotes; tmp; tmp &= tmp - 1) {
        const int bit = __builtin_ctzll(tmp);
        r.midiOut.push({true,  static_cast<uint8_t>(NOTE_BASE + bit), static_cast<uint8_t>(newVel[bit])});
        r.activeNotes |= (1ULL << bit);
    }

    // Decrement holds; OFF expired
    for (uint64_t tmp = r.holdNotes; tmp; tmp &= tmp - 1) {
        const int bit = __builtin_ctzll(tmp);
        if (--r.holdCounts[bit] <= 0) {
            r.midiOut.push({false, static_cast<uint8_t>(NOTE_BASE + bit), 0});
            r.activeNotes &= ~(1ULL << bit);
            r.holdNotes   &= ~(1ULL << bit);
            r.holdCounts[bit] = 0;
        }
    }

    // Active notes absent from newBits and not in hold → start hold or OFF
    for (uint64_t tmp = r.activeNotes & ~newBits & ~r.holdNotes; tmp; tmp &= tmp - 1) {
        const int bit = __builtin_ctzll(tmp);
        // A cancelled/corrected OBP provisional was never CNN-confirmed — skip hold.
        const bool cancelledProv = (prov != -1 && (NOTE_BASE + bit) == prov
                                    && !bmTest(newBits, prov));
        if (r.cfg.holdCycles > 0 && !cancelledProv) {
            r.holdNotes      |= (1ULL << bit);
            r.holdCounts[bit]  = static_cast<int8_t>(r.cfg.holdCycles);
        } else {
            r.midiOut.push({false, static_cast<uint8_t>(NOTE_BASE + bit), 0});
            r.activeNotes &= ~(1ULL << bit);
        }
    }
}

static void runWorker(NeuralNotePlugin* self)
{
    std::array<bool, 8> hasPriorResult = {};

    while (true) {
        sem_wait(&self->workerSem);
        if (self->workerQuit.load(std::memory_order_acquire)) break;

        const float ampFloor = self->ampFloorVal.load(std::memory_order_relaxed);

        for (int ri = 0; ri < static_cast<int>(self->ranges.size()); ++ri) {
            PerRangeState& r = *self->ranges[ri];
            const bool hasSnap    = r.snapChan.ready.load(std::memory_order_acquire);
            const bool paramsChgd = r.paramsChanged.exchange(false, std::memory_order_acq_rel);

            if (self->singleRangeMode) {
                const float ft       = r.frameThresholdVal.load(std::memory_order_relaxed);
                const float os       = r.onsetSensitivity.load(std::memory_order_relaxed);
                const float frames   = r.minNoteFramesVal.load(std::memory_order_relaxed);
                const float minDurMs = frames * FFT_HOP / AUDIO_SAMPLE_RATE * 1000.0f;
                r.basicPitch->setParameters(1.0f - ft, os, minDurMs);
            } else {
                const float minDurMs = r.cfg.minNoteLength * FFT_HOP / AUDIO_SAMPLE_RATE * 1000.0f;
                r.basicPitch->setParameters(1.0f - r.cfg.frameThreshold,
                                             r.cfg.threshold, minDurMs);
            }

            if (hasSnap) {
                r.basicPitch->transcribeToMIDI(r.snapChan.data.data(),
                                                r.snapChan.snapshotSize);
                r.snapChan.ready.store(false, std::memory_order_release);
                hasPriorResult[ri] = true;

                // Two-phase: insert provisional before diff.
                // Always clear so stale out-of-range values never repeat.
                const int prov = r.snapChan.provNoteAtDispatch;
                r.snapChan.provNoteAtDispatch = -1;
                if (prov != -1) {
                    r.provNote.store(-1, std::memory_order_release);
                    if (prov >= NOTE_BASE && prov < NOTE_BASE + NOTE_COUNT) {
                        r.activeNotes |= (1ULL << (prov - NOTE_BASE));
                    } else {
                        // Out-of-bitmap provisional: send OFF immediately.
                        r.midiOut.push({false, static_cast<uint8_t>(prov), 0});
                    }
                }
                // Pass -1 for out-of-bitmap prov — already handled above
                const int provForDiff = (prov >= NOTE_BASE && prov < NOTE_BASE + NOTE_COUNT)
                                        ? prov : -1;
                applyRangeDiff(r, ampFloor, provForDiff);
            } else if (paramsChgd && hasPriorResult[ri] && self->singleRangeMode) {
                r.basicPitch->updateMIDI();
                applyRangeDiff(r, ampFloor);
            }
        }
    }

    // Shutdown: note-offs for all active notes across all ranges
    for (auto& rp : self->ranges) {
        for (uint64_t tmp = rp->activeNotes; tmp; tmp &= tmp - 1)
            rp->midiOut.push({false, static_cast<uint8_t>(NOTE_BASE + __builtin_ctzll(tmp)), 0});
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

static void pushToRing(PerRangeState& r, double sampleRate,
                        const float* in, int inLen)
{
    const int rs = r.ringSize;
    if (sampleRate == 22050.0) {
        for (int i = 0; i < inLen; ++i) {
            r.ringBuf[r.ringHead] = in[i];
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
        r.ringBuf[r.ringHead] =
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

    std::string cfgPath = std::string(bundlePath);
    if (!cfgPath.empty() && cfgPath.back() != '/') cfgPath += '/';
    cfgPath += "neuralnote_ranges.conf";
    RangeConfig rangeCfg = loadRangeConfig(cfgPath);

    auto makeRange = [&](const NoteRange& cfg) {
        auto r             = std::make_unique<PerRangeState>();
        r->cfg             = cfg;
        r->ringSize        = windowMsToRingSize(cfg.windowMs);
        r->minFreshSamples = std::max(r->ringSize / 2, MIN_FRESH_FLOOR);
        r->ringBuf.assign(RING_MAX, 0.0f);
        r->basicPitch = std::make_unique<BasicPitch>();
        const float minDurMs = cfg.minNoteLength * FFT_HOP / AUDIO_SAMPLE_RATE * 1000.0f;
        r->basicPitch->setParameters(1.0f - cfg.frameThreshold, cfg.threshold, minDurMs);
        // Configure per-range OBP lowpass
        const float sr     = static_cast<float>(rate);
        const float cutoff = std::min(440.0f * std::pow(2.0f, (cfg.midiHigh - 69) / 12.0f) * 1.5f,
                                      sr * 0.45f);
        r->obd.setLowpass(cutoff, sr);
#ifdef NEURALNOTE_ENABLE_MPM
        r->mpm.init(sr, cfg.midiLow, cfg.midiHigh);
#endif
        return r;
    };

    if (!rangeCfg.ranges.empty()) {
        self->singleRangeMode = false;
        for (const auto& rc : rangeCfg.ranges)
            self->ranges.push_back(makeRange(rc));
        self->ampFloorVal.store(rangeCfg.ampFloor, std::memory_order_relaxed);
        lv2_log_note(&self->logger, "NeuralNote: loaded %zu ranges from %s\n",
                     self->ranges.size(), cfgPath.c_str());
    } else {
        self->singleRangeMode = true;
        NoteRange def;
        def.midiLow = 0; def.midiHigh = 127; def.name = "default";
        def.windowMs = 150.0f; def.threshold = 0.6f; def.frameThreshold = 0.5f;
        def.minNoteLength = 6; def.holdCycles = 2;
        auto r = makeRange(def);
        r->onsetSensitivity.store(0.6f, std::memory_order_relaxed);
        r->frameThresholdVal.store(0.5f, std::memory_order_relaxed);
        r->minNoteFramesVal.store(6.0f, std::memory_order_relaxed);
        self->ranges.push_back(std::move(r));
    }

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
        case PORT_AUDIO_OUT:       self->audioOut           = static_cast<float*>(data);             break;
        case PORT_MIDI_OUT:        self->midiOut            = static_cast<LV2_Atom_Sequence*>(data); break;
        case PORT_THRESHOLD:       self->threshold          = static_cast<const float*>(data);       break;
        case PORT_MODE:            self->windowMs           = static_cast<const float*>(data);       break;
        case PORT_GATE:            self->gate               = static_cast<const float*>(data);       break;
        case PORT_MIN_DUR:         self->minDur             = static_cast<const float*>(data);       break;
        case PORT_AMP_FLOOR:       self->ampFloor           = static_cast<const float*>(data);       break;
        case PORT_FRAME_THRESHOLD: self->frameThresholdPort = static_cast<const float*>(data);       break;
        case PORT_MIN_NOTE_LENGTH: self->minNoteLengthPort  = static_cast<const float*>(data);       break;
    }
}

static void activate(LV2_Handle instance)
{
    NeuralNotePlugin* self = static_cast<NeuralNotePlugin*>(instance);
    for (auto& rp : self->ranges) {
        rp->ringHead     = 0;
        rp->ringFilled   = 0;
        rp->freshSamples = 0;
        std::fill(rp->ringBuf.begin(), rp->ringBuf.end(), 0.0f);
        rp->basicPitch->reset();
        rp->obd.reset();
        rp->obdVoting.reset();
#ifdef NEURALNOTE_ENABLE_MPM
        rp->mpm.reset();
#endif
        rp->obdOnsetActive  = false;
        rp->obdWindowRemain = 0;
        rp->provNote.store(-1, std::memory_order_relaxed);
        rp->snapChan.provNoteAtDispatch = -1;
        // Send note-offs for any active notes
        for (uint64_t tmp = rp->activeNotes; tmp; tmp &= tmp - 1)
            rp->midiOut.push({false, static_cast<uint8_t>(NOTE_BASE + __builtin_ctzll(tmp)), 0});
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
                      pn.pitch,
                      pn.noteOn ? pn.velocity : uint8_t(0));
    }

    // Single-range mode: propagate LV2 port changes to worker atomics
    if (self->singleRangeMode) {
        PerRangeState& r     = *self->ranges[0];
        bool           chgd  = false;

        if (self->threshold) {
            const float v = *self->threshold;
            if (v != self->lastThreshold) { self->lastThreshold = v; chgd = true; }
            r.onsetSensitivity.store(v, std::memory_order_relaxed);
        }
        if (self->frameThresholdPort) {
            const float v = *self->frameThresholdPort;
            if (v != self->lastFrameThreshold) { self->lastFrameThreshold = v; chgd = true; }
            r.frameThresholdVal.store(v, std::memory_order_relaxed);
        }
        if (self->minNoteLengthPort) {
            const float v = *self->minNoteLengthPort;
            if (v != self->lastMinNoteFrames) { self->lastMinNoteFrames = v; chgd = true; }
            r.minNoteFramesVal.store(v, std::memory_order_relaxed);
        }
        if (self->minDur) {
            const float v = *self->minDur;
            if (v != self->lastMinDur) { self->lastMinDur = v; chgd = true; }
        }
        if (chgd) {
            r.paramsChanged.store(true, std::memory_order_release);
            sem_post(&self->workerSem);
        }

        // Window size change: reset ring + cancel pending snapshot
        if (self->windowMs) {
            const int newRingSize = windowMsToRingSize(*self->windowMs);
            if (newRingSize != r.ringSize) {
                r.snapChan.ready.store(false, std::memory_order_release);
                r.ringSize         = newRingSize;
                r.minFreshSamples  = std::max(newRingSize / 2, MIN_FRESH_FLOOR);
                r.ringHead    = 0;
                r.ringFilled  = 0;
                r.freshSamples = 0;
                std::fill(r.ringBuf.begin(), r.ringBuf.end(), 0.0f);
                r.basicPitch->reset();
                // Release note-offs directly into the output queue
                for (uint64_t tmp = r.activeNotes; tmp; tmp &= tmp - 1)
                    r.midiOut.push({false, static_cast<uint8_t>(NOTE_BASE + __builtin_ctzll(tmp)), 0});
                r.activeNotes = 0;
                r.holdNotes   = 0;
            }
        }
    }

    // amp_floor global update
    if (self->ampFloor) {
        const float v = *self->ampFloor;
        if (v != self->lastAmpFloor) {
            self->lastAmpFloor = v;
            self->ampFloorVal.store(v, std::memory_order_relaxed);
        }
    }

    if (self->audioOut)
        std::memset(self->audioOut, 0, nSamples * sizeof(float));

    // Noise gate + onset detection (share one RMS computation)
    const float gateFloor = (self->gate && *self->gate > 0.0f) ? *self->gate : 0.003f;
    float sumSq = 0.0f;
    for (uint32_t i = 0; i < nSamples; ++i)
        sumSq += self->audioIn[i] * self->audioIn[i];
    const float blockRms = std::sqrt(sumSq / static_cast<float>(nSamples));
    const bool  gated    = (blockRms < gateFloor);

    // Onset: fire a force-dispatch when the block RMS jumps above ONSET_RATIO × background.
    // A 50 ms blank prevents re-triggering on the same note.
    bool onsetFired = false;
    if (self->onsetBlankRemain > 0) {
        self->onsetBlankRemain -= static_cast<int>(nSamples);
        if (self->onsetBlankRemain < 0) self->onsetBlankRemain = 0;
    } else if (!gated && blockRms > self->onsetSmoothedRms * ONSET_RATIO) {
        onsetFired             = true;
        self->onsetBlankRemain = static_cast<int>(self->sampleRate * (ONSET_BLANK_MS / 1000.0f));
        self->onsetSmoothedRms = blockRms; // jump to current level to suppress immediate re-trigger
    }
    if (!onsetFired && self->onsetBlankRemain == 0)
        self->onsetSmoothedRms = self->onsetSmoothedRms * (1.0f - ONSET_ALPHA) + blockRms * ONSET_ALPHA;

    static const float zeros[8192] = {};

    // Push audio into each range's ring; dispatch snapshot when ready — lockless
    for (auto& rp : self->ranges) {
        PerRangeState& r = *rp;

        if (!gated) {
            pushToRing(r, self->sampleRate, self->audioIn, static_cast<int>(nSamples));

            // Two-phase OneBitPitch + MPM: arm on onset, expire after 100 ms.
            armOrExpireOBP(r, static_cast<float>(self->sampleRate),
                           static_cast<int>(nSamples), onsetFired);

            if (r.obdOnsetActive) {
#ifdef NEURALNOTE_ENABLE_MPM
                // Accumulate native-SR samples into MPM before the OBP sub-block loop
                // so the full callback is buffered when OBP votes below.
                r.mpm.push(self->audioIn, static_cast<int>(nSamples));
#endif
                if (r.provNote.load(std::memory_order_relaxed) == -1) {
                    const int finalNote = runOBPHPS(r, self->audioIn,
                                                    static_cast<int>(nSamples),
                                                    static_cast<float>(self->sampleRate),
                                                    self->ranges);
                    if (finalNote != -1) {
#ifdef NEURALNOTE_ENABLE_MPM
                        // Layer 3: MPM consensus check (Pi 5 only).
                        // Provisional fires only if OBP + HPS + MPM all agree.
                        const int mpmNote = r.mpm.analyze(
                            static_cast<float>(self->sampleRate),
                            r.cfg.midiLow, r.cfg.midiHigh);
                        if (mpmNote != -1 && mpmNote == finalNote) {
                            writeMidi(&self->forge, 0, self->uris.midi_MidiEvent,
                                      0x90, static_cast<uint8_t>(finalNote), uint8_t(100));
                            r.provNote.store(finalNote, std::memory_order_release);
                        }
                        // Silent suppression on disagreement
#else
                        // Pi 4: OBP + HPS only — no MPM available
                        writeMidi(&self->forge, 0, self->uris.midi_MidiEvent,
                                  0x90, static_cast<uint8_t>(finalNote), uint8_t(100));
                        r.provNote.store(finalNote, std::memory_order_release);
#endif
                    }
                }
            }
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
        dispatchSnapshotIfReady(r, onsetFired, r.ringBuf.data(), 0.0, self->workerSem);
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
