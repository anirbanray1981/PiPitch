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

#define PLUGIN_URI "https://github.com/DamRsn/NeuralNote/guitar2midi"

#ifndef NEURALNOTE_IMPL_NAME
#define NEURALNOTE_IMPL_NAME "neuralnote_impl"
#endif

#define NEURALNOTE_STRINGIFY2(x) #x
#define NEURALNOTE_STRINGIFY(x)  NEURALNOTE_STRINGIFY2(x)

// ── Constants ─────────────────────────────────────────────────────────────────

static constexpr int RING_MAX = static_cast<int>(AUDIO_SAMPLE_RATE * 2.0); // 44100

// Absolute floor on fresh samples before a normal (non-onset) dispatch (~25 ms).
// The per-range steady-state threshold is ringSize/2; this floor only applies to
// very short windows so onset-triggered dispatches aren't immediately re-dispatched.
static constexpr int MIN_FRESH_FLOOR = static_cast<int>(AUDIO_SAMPLE_RATE * 0.025f);

// Onset detection: force-dispatch when a pick attack is detected.
// onsetRatio: current block RMS must exceed the smoothed background by this factor.
// onsetAlpha: background tracker time constant (~1/alpha blocks to settle).
// onsetBlankMs: suppression window after a trigger to avoid re-firing on the same note.
static constexpr float ONSET_RATIO    = 3.0f;
static constexpr float ONSET_ALPHA    = 0.05f;
static constexpr float ONSET_BLANK_MS = 50.0f;

// MIDI output queue capacity per range (events between run() calls; 64 is ample).
static constexpr int MIDI_QUEUE_CAP = 64;

static inline int windowMsToRingSize(float ms)
{
    const float clamped = std::clamp(ms, 35.0f, 2000.0f);
    return std::min(static_cast<int>(clamped / 1000.0f * AUDIO_SAMPLE_RATE), RING_MAX);
}

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

// ── Lockless SPSC snapshot channel (run() → worker) ──────────────────────────
//
// Ordering contract:
//   Producer: write data/snapshotSize → ready.store(true, release) → sem_post
//   Consumer: sem_wait → ready.load(acquire) → read data/snapshotSize → ready.store(false, release)
//
// The release/acquire on `ready` provides the happens-before edge; sem_post/wait
// additionally ensures the worker sleeps when idle.

struct SnapshotChannel {
    std::vector<float> data;
    int                snapshotSize       = 0;
    int                provNoteAtDispatch = -1;  // set by run() at dispatch; read by worker
    std::atomic<bool>  ready{false};
    std::atomic<bool>  quit{false};
    sem_t              sem;

    SnapshotChannel()  { data.resize(RING_MAX); sem_init(&sem, 0, 0); }
    ~SnapshotChannel() { sem_destroy(&sem); }
    SnapshotChannel(const SnapshotChannel&)          = delete;
    SnapshotChannel& operator=(const SnapshotChannel&) = delete;
    SnapshotChannel(SnapshotChannel&&)               = delete;
    SnapshotChannel& operator=(SnapshotChannel&&)    = delete;
};

// ── Lockless SPSC MIDI output queue (worker → run()) ─────────────────────────
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
    std::set<uint8_t>      activeSet;
    std::map<uint8_t, int> noteHold;

    // Two-phase pitch detection: OneBitPitch fires a provisional note-ON
    // immediately; BasicPitch CNN confirms, corrects, or cancels it.
    OneBitPitchDetector obd;
    std::atomic<int>    provNote{-1};  // -1 = none; set by run(), cleared by worker
    OBPVotingBuffer     obdVoting;
    bool                obdOnsetActive  = false;
    int                 obdWindowRemain = 0;

    // Per-range worker thread
    std::thread workerThread;

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
    int   onsetBlankRemain = 0;      // raw input samples remaining in blank period
};

// ── Worker thread (one per range) ─────────────────────────────────────────────

static void applyRangeDiff(PerRangeState& r, float ampFloor)
{
    std::set<uint8_t>          newSet;
    std::map<uint8_t, uint8_t> velMap;

    for (const auto& ev : r.basicPitch->getNoteEvents()) {
        if (static_cast<float>(ev.amplitude) < ampFloor) continue;
        const int p = static_cast<int>(ev.pitch);
        if (p < r.cfg.midiLow || p > r.cfg.midiHigh) continue;
        const auto pitch = static_cast<uint8_t>(std::clamp(p, 0, 127));
        const auto vel   = static_cast<uint8_t>(
            std::clamp(static_cast<int>(ev.amplitude * 127.0), 1, 127));
        newSet.insert(pitch);
        auto it = velMap.find(pitch);
        if (it == velMap.end() || vel > it->second) velMap[pitch] = vel;
    }

    for (auto p : newSet) {
        r.noteHold.erase(p);
        if (!r.activeSet.count(p)) {
            r.midiOut.push({true, p, velMap[p]});
            r.activeSet.insert(p);
        }
    }

    for (auto it = r.noteHold.begin(); it != r.noteHold.end(); ) {
        if (--(it->second) <= 0) {
            r.midiOut.push({false, it->first, 0});
            r.activeSet.erase(it->first);
            it = r.noteHold.erase(it);
        } else {
            ++it;
        }
    }

    std::vector<uint8_t> immediateOff;
    for (auto p : r.activeSet) {
        if (newSet.count(p) || r.noteHold.count(p)) continue;
        if (r.cfg.holdCycles > 0) {
            r.noteHold[p] = r.cfg.holdCycles;
        } else {
            r.midiOut.push({false, p, 0});
            immediateOff.push_back(p);
        }
    }
    for (auto p : immediateOff) r.activeSet.erase(p);
}

static void runWorkerForRange(NeuralNotePlugin* self, PerRangeState* r)
{
    bool hasPriorResult = false;

    while (true) {
        sem_wait(&r->snapChan.sem);

        if (r->snapChan.quit.load(std::memory_order_acquire)) break;

        const float ampFloor     = self->ampFloorVal.load(std::memory_order_relaxed);
        const bool  hasSnap      = r->snapChan.ready.load(std::memory_order_acquire);
        const bool  paramsChgd   = r->paramsChanged.exchange(false, std::memory_order_acq_rel);

        // Apply latest parameters
        if (self->singleRangeMode) {
            const float ft       = r->frameThresholdVal.load(std::memory_order_relaxed);
            const float os       = r->onsetSensitivity.load(std::memory_order_relaxed);
            const float frames   = r->minNoteFramesVal.load(std::memory_order_relaxed);
            const float minDurMs = frames * FFT_HOP / AUDIO_SAMPLE_RATE * 1000.0f;
            r->basicPitch->setParameters(1.0f - ft, os, minDurMs);
        } else {
            const float minDurMs = r->cfg.minNoteLength * FFT_HOP / AUDIO_SAMPLE_RATE * 1000.0f;
            r->basicPitch->setParameters(1.0f - r->cfg.frameThreshold,
                                          r->cfg.threshold, minDurMs);
        }

        if (hasSnap) {
            r->basicPitch->transcribeToMIDI(r->snapChan.data.data(),
                                             r->snapChan.snapshotSize);
            // Release the slot so run() can write the next snapshot
            r->snapChan.ready.store(false, std::memory_order_release);
            hasPriorResult = true;

            // Two-phase: if a provisional OneBitPitch note was sent, insert it into
            // activeSet before applyRangeDiff so the CNN result handles it cleanly:
            //   CNN confirms  → note in both activeSet + newSet → no double note-ON
            //   CNN corrects  → applyRangeDiff sends note-OFF + correct note-ON
            //   CNN finds nil → applyRangeDiff sends note-OFF
            const int prov = r->snapChan.provNoteAtDispatch;
            if (prov != -1) {
                r->activeSet.insert(static_cast<uint8_t>(prov));
                r->snapChan.provNoteAtDispatch = -1;
                r->provNote.store(-1, std::memory_order_release);
            }

            applyRangeDiff(*r, ampFloor);
        } else if (paramsChgd && hasPriorResult && self->singleRangeMode) {
            r->basicPitch->updateMIDI();   // fast re-evaluation, no CNN
            applyRangeDiff(*r, ampFloor);
        }
    }

    // Shutdown: push note-offs for all active notes
    for (auto p : r->activeSet)
        r->midiOut.push({false, p, 0});
    r->activeSet.clear();
    r->noteHold.clear();
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

    for (auto& rp : self->ranges)
        rp->workerThread = std::thread(runWorkerForRange, self, rp.get());

    lv2_log_note(&self->logger,
                 "NeuralNote Guitar2MIDI: %.0f Hz  [impl: " NEURALNOTE_IMPL_NAME
                 "]  [ranges: %zu, lockless workers]\n", rate, self->ranges.size());
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
        rp->obdOnsetActive  = false;
        rp->obdWindowRemain = 0;
        rp->provNote.store(-1, std::memory_order_relaxed);
        rp->snapChan.provNoteAtDispatch = -1;
        // Send note-offs for any active notes
        for (auto p : rp->activeSet)
            rp->midiOut.push({false, p, 0});
        rp->activeSet.clear();
        rp->noteHold.clear();
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
            sem_post(&r.snapChan.sem);
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
                for (auto p : r.activeSet)
                    r.midiOut.push({false, p, 0});
                r.activeSet.clear();
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

            // Two-phase OneBitPitch: arm on onset, expire after 100 ms.
            // Voting buffer requires 9/12 consistent readings before firing.
            if (onsetFired) {
                r.obdOnsetActive  = true;
                r.obdWindowRemain = static_cast<int>(self->sampleRate * 0.1f);
                r.obdVoting.reset();
                r.obd.reset();
            } else if (r.obdWindowRemain > 0) {
                r.obdWindowRemain -= static_cast<int>(nSamples);
                if (r.obdWindowRemain <= 0) {
                    r.obdWindowRemain = 0;
                    r.obdOnsetActive  = false;
                }
            }
            if (r.obdOnsetActive && r.provNote.load(std::memory_order_relaxed) == -1) {
                const int op = r.obd.process(self->audioIn, static_cast<int>(nSamples),
                                              static_cast<float>(self->sampleRate));
                const int voted = r.obdVoting.update(
                    (op >= r.cfg.midiLow && op <= r.cfg.midiHigh) ? op : -1);
                if (voted != -1) {
                    // Suppress if another range already has a provisional that is
                    // the fundamental of this note (voted - 12 or - 24).
                    bool isHarmonic = false;
                    for (const auto& other : self->ranges) {
                        if (other.get() == &r) continue;
                        const int op2 = other->provNote.load(std::memory_order_relaxed);
                        if (op2 != -1) {
                            const int diff = voted - op2;
                            if (diff == 12 || diff == 24) { isHarmonic = true; break; }
                        }
                    }
                    r.obdOnsetActive  = false;
                    r.obdWindowRemain = 0;
                    r.obdVoting.reset();
                    if (!isHarmonic) {
                        writeMidi(&self->forge, 0, self->uris.midi_MidiEvent,
                                  0x90, static_cast<uint8_t>(voted), uint8_t(100));
                        r.provNote.store(voted, std::memory_order_release);
                    }
                }
            }
        } else {
            // Silence: reset OBP state so stale periods don't persist into next note
            r.obd.reset();
            r.obdVoting.reset();
            r.obdOnsetActive  = false;
            r.obdWindowRemain = 0;
            int rem = static_cast<int>(nSamples);
            while (rem > 0) {
                const int chunk = std::min(rem, 8192);
                pushToRing(r, self->sampleRate, zeros, chunk);
                rem -= chunk;
            }
        }

        // Dispatch snapshot if: ring full, slot free, and either enough fresh audio
        // has arrived (normal path: ringSize/2) or an onset was detected (early path).
        if (!r.snapChan.ready.load(std::memory_order_acquire)
            && r.ringFilled >= r.ringSize
            && (r.freshSamples >= r.minFreshSamples || onsetFired))
        {
            const int rs = r.ringSize;
            for (int i = 0; i < rs; ++i)
                r.snapChan.data[i] = r.ringBuf[(r.ringHead + i) % rs];
            r.snapChan.snapshotSize       = rs;
            r.snapChan.provNoteAtDispatch = r.provNote.load(std::memory_order_relaxed);
            r.snapChan.ready.store(true, std::memory_order_release);
            r.freshSamples = 0;
            sem_post(&r.snapChan.sem);
        }
    }

    lv2_atom_forge_pop(&self->forge, &seqFrame);
}

static void deactivate(LV2_Handle /*instance*/) {}

static void cleanup(LV2_Handle instance)
{
    NeuralNotePlugin* self = static_cast<NeuralNotePlugin*>(instance);
    for (auto& rp : self->ranges) {
        rp->snapChan.quit.store(true, std::memory_order_release);
        sem_post(&rp->snapChan.sem);
    }
    for (auto& rp : self->ranges)
        if (rp->workerThread.joinable())
            rp->workerThread.join();
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
