/**
 * PiPitch — LV2 Implementation (low-latency streaming)
 *
 * Compiled multiple times with different -march flags by CMake.
 * Loaded by pipitch.so (the wrapper) via dlopen().
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
 * If pipitch_ranges.conf exists in the bundle directory each [range] section
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
// Must be defined before PiPitchShared.h so that RangeStateBase and all
// shared pipeline functions compile in the McLeod call-sites.
// The neon (Pi 4) build leaves this undefined — no FFTW dependency.
#if defined(__aarch64__) && defined(__ARM_FEATURE_DOTPROD)
#  define PIPITCH_ENABLE_MPM 1
#endif

#include "BasicPitchConstants.h"
#include "PiPitchShared.h"  // pulls in BinaryData.h, BasicPitch.h, NoteRangeConfig.h,
                               // OneBitPitchDetector.h, McLeodPitchDetector.h (if MPM)
#include "SwiftF0Detector.h"

#define PLUGIN_URI "https://github.com/anirbanray1981/PiPitch"

#ifndef PIPITCH_IMPL_NAME
#define PIPITCH_IMPL_NAME "neuralnote_impl"
#endif

#define PIPITCH_STRINGIFY2(x) #x
#define PIPITCH_STRINGIFY(x)  PIPITCH_STRINGIFY2(x)

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
    PORT_PROVISIONAL     = 9,
    PORT_BEND            = 10,
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
struct PiPitchPlugin;

// ── Per-range runtime state ───────────────────────────────────────────────────
// All common fields (including MidiOutQueue midiOut) live in RangeStateBase.

struct RangeState : RangeStateBase {
};

// ── Plugin instance ───────────────────────────────────────────────────────────

struct PiPitchPlugin {
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
    const float*       provisionalPort;
    const float*       bendPort;

    double sampleRate;

    std::atomic<float> thresholdVal{0.6f};
    std::atomic<float> frameThresholdVal{0.5f};
    std::atomic<float> ampFloorVal{0.65f};
    std::atomic<int>   modeVal{1};  // 0=poly, 1=mono, 2=swiftmono, 3=swiftpoly (default: mono)
    std::atomic<int>   provisionalVal{0};  // 0=on, 1=swift, 2=none
    std::atomic<bool>  bendVal{false};
    float              octaveLockMsVal = 250.0f;  // from config (not an LV2 port)

    std::unique_ptr<SwiftF0Detector> swiftF0;  // null if model not found
#ifdef __aarch64__
    UltraLowLatencyGoertzel          goertzel; // GoertzelPoly mode (audio-thread, zero-latency)
    uint64_t goertzelPrevBits = 0;
#endif
    std::vector<float>               sf0Buf;   // worker-thread scratch: 16 kHz resampled audio

    std::vector<std::unique_ptr<RangeState>> ranges;
    bool singleRangeMode = true;

    // Onset detector state (run() thread only — no synchronisation needed)
    float onsetSmoothedRms = 0.001f;
    int   onsetBlankRemain = 0;

    // HPF pick detector (run() thread only)
    PickDetector pickDetector;
    int          pickFiredRemain = 0;  // samples remaining since last PICK; >0 = recent pick
    float        lastPickRatio   = 0.0f;  // ratio of the most recent PICK fire

    // Onset-recency tracking (audio → worker, for decay-tail ghost suppression)
    std::atomic<uint64_t> totalSamples{0};
    std::atomic<uint64_t> lastOnsetSample{0};

    // Single worker thread shared across all ranges
    std::thread       workerThread;
    std::atomic<bool> workerQuit{false};
    sem_t             workerSem;
};

// ── Worker thread hooks (LV2 plugin) ──────────────────────────────────────────

struct ImplWorkerHooks {
    PiPitchPlugin* self;

    sem_t&              workerSem()      { return self->workerSem; }
    bool                shouldQuit()     { return self->workerQuit.load(std::memory_order_acquire); }
    float               ampFloor()       { return self->ampFloorVal.load(std::memory_order_relaxed); }
    int                 mode()           { return self->modeVal.load(std::memory_order_relaxed); }
    float               frameThreshold() { return self->frameThresholdVal.load(std::memory_order_relaxed); }
    float               threshold()      { return self->thresholdVal.load(std::memory_order_relaxed); }
    float               swiftThreshold() { return 0.5f; }
    double              sampleRate()     { return self->sampleRate; }
    SwiftF0Detector*    swiftF0()        { return self->swiftF0.get(); }
    std::vector<float>& sf0Buf()         { return self->sf0Buf; }
    uint64_t            totalSamples()   { return self->totalSamples.load(std::memory_order_relaxed); }
    uint64_t            lastOnsetSample(){ return self->lastOnsetSample.load(std::memory_order_acquire); }
    int                 provisionalMode(){ return self->provisionalVal.load(std::memory_order_relaxed); }
    float               octaveLockMs()   { return self->octaveLockMsVal; }
    bool                bendEnabled()    { return self->bendVal.load(std::memory_order_relaxed); }
    auto&               ranges()         { return self->ranges; }

    // No logging in LV2 plugin
    void onSwiftResult    (RangeState&, int, double) {}
    void onSwiftPolyResult(RangeState&, int, double, uint64_t, double) {}
    void onCNNOutcome     (RangeState&, int, uint64_t, double) {}
    void onNotesChanged   (RangeState&, uint64_t, const int8_t*, double, const char*) {}
    void onMonoKill       (RangeState&, int) {}
    void onShutdownOff    (RangeState&, int) {}
};

// ── Worker thread ─────────────────────────────────────────────────────────────

static void runWorker(PiPitchPlugin* self)
{
    ImplWorkerHooks hooks{self};
    runWorkerCommon(hooks);
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
    PiPitchPlugin* self = new PiPitchPlugin();
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
    cfgPath += "pipitch_ranges.conf";
    RangeConfig rangeCfg = loadRangeConfig(cfgPath);
    self->octaveLockMsVal = rangeCfg.octaveLockMs;

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
#ifdef PIPITCH_ENABLE_MPM
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

    self->pickDetector.init(static_cast<float>(rate), 3000.0f, 3.0f);
#ifdef __aarch64__
    self->goertzel.init(static_cast<float>(rate), NOTE_BASE, NOTE_BASE + NOTE_COUNT - 1);
#endif

    sem_init(&self->workerSem, 0, 0);
    self->workerThread = std::thread(runWorker, self);

    lv2_log_note(&self->logger,
                 "PiPitch: %.0f Hz  [impl: " PIPITCH_IMPL_NAME
                 "]  [ranges: %zu, single worker thread]\n", rate, self->ranges.size());
    return static_cast<LV2_Handle>(self);
}

static void connectPort(LV2_Handle instance, uint32_t port, void* data)
{
    PiPitchPlugin* self = static_cast<PiPitchPlugin*>(instance);
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
        case PORT_PROVISIONAL:    self->provisionalPort    = static_cast<const float*>(data);       break;
        case PORT_BEND:           self->bendPort           = static_cast<const float*>(data);       break;
    }
}

static void activate(LV2_Handle instance)
{
    PiPitchPlugin* self = static_cast<PiPitchPlugin*>(instance);
    for (auto& rp : self->ranges) {
        rp->ringHead     = 0;
        rp->ringFilled   = 0;
        rp->freshSamples = 0;
        std::fill(rp->ring.begin(), rp->ring.end(), 0.0f);
        rp->basicPitch->reset();
        rp->obd.reset();
        rp->obdVoting.reset();
#ifdef PIPITCH_ENABLE_MPM
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
    PiPitchPlugin* self = static_cast<PiPitchPlugin*>(instance);

    lv2_atom_forge_set_buffer(&self->forge,
                               reinterpret_cast<uint8_t*>(self->midiOut),
                               self->midiOut->atom.size);
    LV2_Atom_Forge_Frame seqFrame;
    lv2_atom_forge_sequence_head(&self->forge, &seqFrame, 0);

    // Drain all per-range MIDI output queues — lockless
    for (auto& rp : self->ranges) {
        PendingNote pn;
        while (rp->midiOut.pop(pn)) {
            if (pn.type == PendingNote::PITCH_BEND) {
                const uint8_t lsb = static_cast<uint8_t>(pn.value & 0x7F);
                const uint8_t msb = static_cast<uint8_t>((pn.value >> 7) & 0x7F);
                writeMidi(&self->forge, 0, self->uris.midi_MidiEvent,
                          uint8_t(0xE0), lsb, msb);
            } else {
                writeMidi(&self->forge, 0, self->uris.midi_MidiEvent,
                          pn.type == PendingNote::NOTE_ON ? uint8_t(0x90) : uint8_t(0x80),
                          static_cast<uint8_t>(pn.pitch),
                          pn.type == PendingNote::NOTE_ON ? static_cast<uint8_t>(pn.value) : uint8_t(0));
                // Clear pending provisional if worker already sent this note
                if (pn.type == PendingNote::NOTE_ON && rp->pendingProvNote == pn.pitch)
                    rp->pendingProvNote = -1;
            }
        }
    }

    // Propagate LV2 port values to worker atomics every callback.
    // No change-detection: the worker must always see the host's current value,
    // including values restored from a preset or snapshot before the first run().
    if (self->thresholdPort)      self->thresholdVal.store(*self->thresholdPort, std::memory_order_relaxed);
    if (self->frameThresholdPort) self->frameThresholdVal.store(*self->frameThresholdPort, std::memory_order_relaxed);
    if (self->modePort)           self->modeVal.store(static_cast<int>(*self->modePort + 0.5f), std::memory_order_relaxed);
    if (self->provisionalPort)    self->provisionalVal.store(static_cast<int>(*self->provisionalPort + 0.5f), std::memory_order_relaxed);
    if (self->bendPort)           self->bendVal.store(*self->bendPort > 0.5f, std::memory_order_relaxed);
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
    if (self->pickFiredRemain > 0)
        self->pickFiredRemain -= static_cast<int>(nSamples);
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
        // Any PICK: reset provisional cooldowns (genuine new attack).
        // The PICK detector's 25ms internal blank prevents double-firing on the
        // same pick's energy, so if PICK fires again it's a real re-hit.
        for (auto& rp2 : self->ranges)
            { rp2->provCooldownRemain = 0; rp2->provCooldownNote = -1; }
        self->pickFiredRemain = static_cast<int>(self->sampleRate * 0.05);  // 50ms window
        self->lastPickRatio   = pickRatio;
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
    // GoertzelPoly bypasses the entire ring/OBP/CNN pipeline.
    const bool goertzelMode = (self->modeVal.load(std::memory_order_relaxed) == 4);
    for (auto& rp : self->ranges) {
        RangeState& r = *rp;
        if (goertzelMode) continue;

        if (!gated) {
            // Flush ring on onset: zero stale audio so SwiftF0 only sees the
            // new note.  Fires on PICK onsets and strong RMS onsets (ratio > 3).
            // Note: don't touch holdNotes/activeNotes — those are worker-owned.
            if (onsetFired) {
                std::memset(r.ring.data(), 0, r.ringSize * sizeof(float));
                r.freshSamples = 0;
            }
            pushToRing(r, self->sampleRate, self->audioIn, static_cast<int>(nSamples));

            // Two-phase OneBitPitch + MPM — skipped when provisional != on.
            // (GoertzelPoly never reaches here — range loop is skipped above.)
            const int pvNow = self->provisionalVal.load(std::memory_order_relaxed);
            const bool provEnabled = (pvNow == 0 || pvNow == 3);  // on or adaptive
            if (provEnabled) {

            // OBP blanking: freeze OBP for 5ms after PICK to skip pick noise.
            // The first 5ms of a pluck is mostly broadband noise that confuses OBP.
            if (onsetFired && self->pickFiredRemain > 0)
                r.obpBlankRemain = static_cast<int>(self->sampleRate * 0.005); // 5ms
            const bool obpBlanked = (r.obpBlankRemain > 0);
            if (obpBlanked)
                r.obpBlankRemain -= static_cast<int>(nSamples);

            armOrExpireOBP(r, static_cast<float>(self->sampleRate),
                           static_cast<int>(nSamples), onsetFired);

            // Decrement provisional cooldown
            if (r.provCooldownRemain > 0)
                r.provCooldownRemain -= static_cast<int>(nSamples);

            // Helper: fire a provisional note — shared by immediate and pending paths.
            auto fireProv = [&](int note) {
                // Skip if same note is under cooldown (RMS re-trigger suppression)
                if (r.provCooldownRemain > 0 && note == r.provCooldownNote)
                    return;
                // Below-C3 filter: in swiftpoly, always discard provisionals
                // below C3 (harmonics from higher notes). In other modes,
                // discard from silence only.
                if (note < MIDI_NOTE_C3) {
                    const int mv = self->modeVal.load(std::memory_order_relaxed);
                    if (mv == 3) return;  // swiftpoly: always discard <C3
                    bool anyActive = false;
                    for (auto& other : self->ranges) {
                        if (other->activeNotesBits.load(std::memory_order_acquire) != 0
                            || other->provNote.load(std::memory_order_acquire) != -1)
                            { anyActive = true; break; }
                    }
                    if (!anyActive) return;  // from silence: discard <C3
                }
                // Octave-lock (cross-range): suppress ±12/±24 semitone provisionals
                // relative to any active note OR recent provisional across all ranges.
                // Checks both activeNotesBits (worker-updated) and provNote (audio-set)
                // since the worker may not have promoted a recent provisional yet.
                if (self->octaveLockMsVal > 0.0f) {
                    for (auto& other : self->ranges) {
                        const uint64_t bits = other->activeNotesBits.load(std::memory_order_acquire);
                        for (uint64_t tmp = bits; tmp; tmp &= tmp - 1) {
                            const int act = NOTE_BASE + static_cast<int>(__builtin_ctzll(tmp));
                            const int diff = std::abs(note - act);
                            if (diff == 12 || diff == 24) return;
                        }
                        const int prov = other->provNote.load(std::memory_order_acquire);
                        if (prov != -1) {
                            const int diff = std::abs(note - prov);
                            if (diff == 12 || diff == 24) return;
                        }
                    }
                }
                // Mono cross-range guard: in mono mode, suppress provisional if
                // any OTHER range already has an active note or pending provisional.
                // Only the first range to fire wins; SwiftF0 corrects later.
                if (self->modeVal.load(std::memory_order_relaxed) >= 1) {
                    for (auto& other : self->ranges) {
                        if (&*other == &r) continue;
                        if (other->activeNotesBits.load(std::memory_order_acquire) != 0)
                            return;
                        if (other->provNote.load(std::memory_order_acquire) != -1)
                            return;
                    }
                }
                // Same note already active: re-hit (OFF+ON) if PICK fired this
                // callback (cooldown was just reset), else skip (RMS re-trigger).
                if (note >= NOTE_BASE && note < NOTE_BASE + NOTE_COUNT) {
                    const uint64_t bits = r.activeNotesBits.load(std::memory_order_acquire);
                    if (bits & (1ULL << (note - NOTE_BASE))) {
                        if (self->pickFiredRemain <= 0) return;  // no recent PICK → skip
                        // E2-B2 range: require stronger PICK for re-hits (avoid
                        // sympathetic resonance triggers). C3+: lower threshold OK.
                        if (note < 48 && self->lastPickRatio < 4.0f) return;
                        // PICK-triggered re-hit: send OFF first for clean retrigger
                        writeMidi(&self->forge, 0, self->uris.midi_MidiEvent,
                                  0x80, static_cast<uint8_t>(note), uint8_t(0));
                    } else if (bits != 0) {
                        // Transition: defer MIDI ON, store for SwiftF0 consensus.
                        // Clear provNote so worker doesn't silently insert into activeNotes.
                        r.transitionProv.store(note, std::memory_order_release);
                        r.provNote.store(-1, std::memory_order_release);
                        r.provCooldownRemain = static_cast<int>(self->sampleRate * 0.2);
                        r.provCooldownNote   = note;
                        return;
                    }
                }
                // Mono / SwiftMono: kill any other range's held note (NOT swiftpoly)
                {   const int mv = self->modeVal.load(std::memory_order_relaxed);
                if (mv == 1 || mv == 2) {
                    for (auto& other : self->ranges) {
                        const int held = other->monoHeldNote.load(std::memory_order_acquire);
                        if (held != -1 && held != note) {
                            writeMidi(&self->forge, 0, self->uris.midi_MidiEvent,
                                      0x80, static_cast<uint8_t>(held), uint8_t(0));
                            other->monoHeldNote.store(-1, std::memory_order_release);
                        }
                    }
                }}
                // Kill existing provisional in this range if different note
                {   const int oldProv = r.provNote.load(std::memory_order_acquire);
                    if (oldProv != -1 && oldProv != note) {
                        writeMidi(&self->forge, 0, self->uris.midi_MidiEvent,
                                  0x80, static_cast<uint8_t>(oldProv), uint8_t(0));
                        // Reset pitch bend if it was snapped
                        if (r.provBentTo >= 0) {
                            writeMidi(&self->forge, 1, self->uris.midi_MidiEvent,
                                      0xE0, uint8_t(0), uint8_t(0x40)); // center
                        }
                    }
                }
                const int pv = self->provisionalVal.load(std::memory_order_relaxed);
                if (pv == 3) {
                    // Adaptive mode: don't send MIDI ON; worker decides after SwiftF0
                    r.provNote.store(note, std::memory_order_release);
                    r.monoHeldNote.store(note, std::memory_order_release);
                } else {
                    // Confirmation buffer: hold provisional for ~10ms.
                    // If the worker pushes a correction within that window,
                    // use it instead. After timeout, fire at muted velocity.
                    r.pendingProvNote = note;
                    r.pendingProvCountdown = static_cast<int>(self->sampleRate * 0.010); // 10ms
                    r.provNote.store(note, std::memory_order_release);
                    r.monoHeldNote.store(note, std::memory_order_release);
                }
                r.provBentTo = -1;
                r.provNeedsBoost = (pv != 3);
                r.provCooldownRemain = static_cast<int>(self->sampleRate * 0.2);
                r.provCooldownNote   = note;
            };

#ifdef PIPITCH_ENABLE_MPM
            // Push to MPM while OBP window is active OR while awaiting MPM on a
            // pending OBP vote — so we can retry analyze() next callback.
            if (r.obdOnsetActive || r.obdPendingNote != -1)
                r.mpm.push(self->audioIn, static_cast<int>(nSamples));
#endif

            if (r.obdOnsetActive && !obpBlanked) {
                if (r.provNote.load(std::memory_order_relaxed) == -1) {
                    const int finalNote = runOBPHPS(r, self->audioIn,
                                                    static_cast<int>(nSamples),
                                                    static_cast<float>(self->sampleRate),
                                                    self->ranges);
                    // MPM fallback: OBP expired with no vote — try MPM alone
                    if (finalNote == -1 && !r.obdOnsetActive) {
#ifdef PIPITCH_ENABLE_MPM
                        const int mpmNote = r.mpm.analyze(
                            static_cast<float>(self->sampleRate),
                            r.cfg.midiLow, r.cfg.midiHigh);
                        if (mpmNote != -1) {
                            const int mv2 = self->modeVal.load(std::memory_order_relaxed);
                            const bool mono = (mv2 == 1 || mv2 == 2);
                            bool suppress = false;
                            if (mono) {
                                for (auto& orp : self->ranges) {
                                    if (&*orp == &r) continue;
                                    if (orp->activeNotesBits.load(std::memory_order_acquire) != 0
                                        || orp->provNote.load(std::memory_order_relaxed) != -1)
                                        { suppress = true; break; }
                                }
                            }
                            if (!suppress) fireProv(mpmNote);
                        }
#endif
                    }
                    if (finalNote != -1) {
                        bool shouldFire = false;
#ifdef PIPITCH_ENABLE_MPM
                        const int mpmNote = r.mpm.analyze(
                            static_cast<float>(self->sampleRate),
                            r.cfg.midiLow, r.cfg.midiHigh);
                        if (mpmNote != -1 && mpmNote == finalNote) {
                            // OBP + MPM agree — fire immediately
                            shouldFire = true;
                        } else if (mpmNote != -1) {
                            // OBP + MPM disagree — save pending; retry with more data
                            // (MPM at low fill can be inaccurate; pending retry trusts MPM)
                            r.obdPendingNote   = finalNote;
                            r.obdPendingRemain = static_cast<int>(self->sampleRate * 0.1f);
                        } else {
                            // MPM not ready — save pending
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
#ifdef PIPITCH_ENABLE_MPM
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
                            if (mono && (self->ranges[oi]->activeNotesBits.load(std::memory_order_acquire) != 0
                                         || self->ranges[oi]->provNote.load(std::memory_order_relaxed) != -1))
                                { suppress = true; break; }
                            const int op2 = self->ranges[oi]->provNote.load(std::memory_order_relaxed);
                            if (op2 != -1) {
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
            } // provEnabled
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

    // ── GoertzelPoly: sample-by-sample processing in audio thread ──────────
    // Only available on AArch64 (Pi 5) — requires NEON SIMD.
#ifdef __aarch64__
    if (self->modeVal.load(std::memory_order_relaxed) == 4 && !gated) {
        self->goertzel.processBlock(self->audioIn, static_cast<int>(nSamples),
                                     onsetFired && self->pickFiredRemain > 0);

        // Check note states using triggerPending (fire-once per onset)
        auto& states = self->goertzel.getNoteStates();
        const int gStart = self->goertzel.startMidi();
        for (int i = 0; i < self->goertzel.numNotes(); ++i) {
            const int midi = gStart + i;
            if (midi < NOTE_BASE || midi >= NOTE_BASE + NOTE_COUNT) continue;
            auto& s = states[i];
            const uint64_t bit = 1ULL << (midi - NOTE_BASE);

            if (s.isActive() && !(self->goertzelPrevBits & bit)) {
                // New note-ON: only fire if triggerPending (prevents re-trigger)
                if (s.triggerPending) {
                    // Octave-lock: suppress if ±12/±24 semitones from any
                    // already-active Goertzel note (harmonic false positive).
                    bool octLocked = false;
                    for (uint64_t tmp = self->goertzelPrevBits; tmp; tmp &= tmp - 1) {
                        const int act = NOTE_BASE + __builtin_ctzll(tmp);
                        const int diff = std::abs(midi - act);
                        if (diff == 12 || diff == 24) { octLocked = true; break; }
                    }
                    if (!octLocked) {
                        writeMidi(&self->forge, 0, self->uris.midi_MidiEvent,
                                  0x90, static_cast<uint8_t>(midi),
                                  static_cast<uint8_t>(s.velocity));
                        self->goertzelPrevBits |= bit;
                    }
                    s.triggerPending = false;  // consume regardless
                }
            } else if (!s.isActive() && (self->goertzelPrevBits & bit)) {
                // Note-OFF
                writeMidi(&self->forge, 0, self->uris.midi_MidiEvent,
                          0x80, static_cast<uint8_t>(midi), uint8_t(0));
                self->goertzelPrevBits &= ~bit;
            }
        }
    } else if (self->modeVal.load(std::memory_order_relaxed) == 4 && gated) {
        // Gated: turn off all active Goertzel notes
        for (uint64_t tmp = self->goertzelPrevBits; tmp; tmp &= tmp - 1) {
            const int p = NOTE_BASE + static_cast<int>(__builtin_ctzll(tmp));
            writeMidi(&self->forge, 0, self->uris.midi_MidiEvent,
                      0x80, static_cast<uint8_t>(p), uint8_t(0));
        }
        self->goertzelPrevBits = 0;
        self->goertzel.reset();
    }
#endif // __aarch64__

    // Second drain: catch any CNN events the worker pushed during this callback.
    // The first drain (top of run) caught events from before this cycle;
    // this one reduces CNN note latency by up to one JACK buffer.
    for (auto& rp : self->ranges) {
        PendingNote pn;
        while (rp->midiOut.pop(pn)) {
            if (pn.type == PendingNote::PITCH_BEND) {
                const uint8_t lsb = static_cast<uint8_t>(pn.value & 0x7F);
                const uint8_t msb = static_cast<uint8_t>((pn.value >> 7) & 0x7F);
                writeMidi(&self->forge, 0, self->uris.midi_MidiEvent,
                          uint8_t(0xE0), lsb, msb);
            } else {
                writeMidi(&self->forge, 0, self->uris.midi_MidiEvent,
                          pn.type == PendingNote::NOTE_ON ? uint8_t(0x90) : uint8_t(0x80),
                          static_cast<uint8_t>(pn.pitch),
                          pn.type == PendingNote::NOTE_ON ? static_cast<uint8_t>(pn.value) : uint8_t(0));
                if (pn.type == PendingNote::NOTE_ON && rp->pendingProvNote == pn.pitch)
                    rp->pendingProvNote = -1;
            }
        }
    }

    // Confirmation buffer: fire buffered provisionals after 10ms timeout.
    // If the worker pushed a correction (via midiOut drain above), the
    // provisional was already superseded. Otherwise fire at muted velocity.
    for (auto& rp : self->ranges) {
        RangeState& r = *rp;
        if (r.pendingProvNote >= 0) {
            r.pendingProvCountdown -= static_cast<int>(nSamples);
            if (r.pendingProvCountdown <= 0) {
                // Timeout: fire the provisional at muted velocity
                writeMidi(&self->forge, 0, self->uris.midi_MidiEvent,
                          0x90, static_cast<uint8_t>(r.pendingProvNote), uint8_t(40));
                r.pendingProvNote = -1;
            }
        }
    }

    lv2_atom_forge_pop(&self->forge, &seqFrame);
}

static void deactivate(LV2_Handle /*instance*/) {}

static void cleanup(LV2_Handle instance)
{
    PiPitchPlugin* self = static_cast<PiPitchPlugin*>(instance);
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

LV2_SYMBOL_EXPORT const LV2_Descriptor* pipitch_impl_descriptor(uint32_t index)
{
    return (index == 0) ? &descriptor : nullptr;
}
