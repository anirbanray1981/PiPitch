/**
 * NeuralNote Guitar2MIDI — LV2 Implementation (low-latency streaming)
 *
 * Compiled multiple times with different -march flags by CMake.
 * Loaded by neuralnote_guitar2midi.so (the wrapper) via dlopen().
 *
 * Latency model
 * ─────────────
 * Previous design: accumulate 2 s → transcribe → clear → repeat.
 *   Latency ≈ 2000 ms (fill) + 378 ms (inference) = ~2400 ms.
 *
 * This design: ring buffer + background worker thread.
 *   A circular buffer always holds the latest 2 s of 22050-Hz audio.
 *   run() writes new audio into the ring and immediately hands a snapshot
 *   to the worker (non-blocking try_to_lock).  The worker runs inference
 *   continuously; as soon as one inference finishes the next snapshot is
 *   already waiting.
 *
 *   Effective latency ≈ inference_time + one_run()_block
 *                     ≈ 380 ms on Pi 4 (NEON)
 *                     ≈ 200 ms on Pi 5 (dotprod+fp16, estimated)
 *
 * Note tracking
 * ─────────────
 * Instead of paired note-on/off from a single batch, we maintain
 * `activeSet`: the set of pitches present in the most recent inference.
 * Each inference result is diffed against activeSet:
 *   new pitches  → note-on
 *   gone pitches → note-off
 * This naturally handles the sliding window without duplicate events.
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
#include <condition_variable>
#include <cstring>
#include <cstdlib>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <thread>
#include <vector>

// BinaryData.h must come before any Lib/Model header
#include "BinaryData.h"
#include "BasicPitch.h"
#include "BasicPitchConstants.h"

#define PLUGIN_URI "https://github.com/DamRsn/NeuralNote/guitar2midi"

#ifndef NEURALNOTE_IMPL_NAME
#define NEURALNOTE_IMPL_NAME "neuralnote_impl"
#endif

#define NEURALNOTE_STRINGIFY2(x) #x
#define NEURALNOTE_STRINGIFY(x)  NEURALNOTE_STRINGIFY2(x)

// ── Constants ─────────────────────────────────────────────────────────────────

// Ring buffer allocated at the maximum window size (Slow = 1000 ms).
// The active window is set at runtime by the Mode control port.
static constexpr int RING_MAX = static_cast<int>(AUDIO_SAMPLE_RATE * 1.0); // 22050

// Map mode port value → ring size in 22050-Hz samples.
//   0 = Fast   (300 ms)  →  6615  — 4× 1/16 note at 120 BPM
//   1 = Medium (500 ms)  → 11025  — 4× 1/16 note at 100 BPM  [default]
//   2 = Slow   (1000 ms) → 22050  — complex chords / slow tempos
static inline int modeToRingSize(float mode)
{
    if (mode < 0.5f) return static_cast<int>(AUDIO_SAMPLE_RATE * 0.3); // 6615
    if (mode < 1.5f) return static_cast<int>(AUDIO_SAMPLE_RATE * 0.5); // 11025
    return RING_MAX;                                                      // 22050
}

// ── Port indices ──────────────────────────────────────────────────────────────

enum PortIndex {
    PORT_AUDIO_IN = 0,
    PORT_MIDI_OUT = 1,
    PORT_THRESHOLD = 2,
    PORT_MODE = 3,
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

// ── Plugin instance ───────────────────────────────────────────────────────────

struct NeuralNotePlugin {
    // LV2 infrastructure
    LV2_URID_Map*   map;
    LV2_Log_Logger  logger;
    LV2_Atom_Forge  forge;
    URIs            uris;

    // Ports
    const float*       audioIn;
    LV2_Atom_Sequence* midiOut;
    const float*       threshold;
    const float*       mode;      // 0=Fast 300ms  1=Medium 500ms  2=Slow 1000ms

    // Inference engine — only accessed by the worker thread after init
    std::unique_ptr<BasicPitch> basicPitch;
    double sampleRate;

    // Parameters: written by run(), read by worker (atomic for safety)
    std::atomic<float> noteSensitivity{0.7f};
    std::atomic<float> splitSensitivity{0.5f};
    std::atomic<float> minNoteLengthMs{50.0f};

    // ── Ring buffer (22050-Hz resampled audio, circular) ──────────────────
    // Allocated at RING_MAX (1000 ms); ringSize tracks the active window.
    // ringHead: next write slot (= oldest sample when ring is full).
    // ringFilled: valid samples, capped at ringSize.
    float ringBuf[RING_MAX];
    int   ringSize   = static_cast<int>(AUDIO_SAMPLE_RATE * 0.5); // default Medium
    int   ringHead   = 0;
    int   ringFilled = 0;

    // ── Background inference worker ────────────────────────────────────────
    std::thread             workerThread;
    std::mutex              workerMutex;
    std::condition_variable workerCv;
    bool                    workerHasWork = false;
    bool                    workerQuit    = false;
    std::vector<float>      workerSnapshot; // ring linearised for the worker

    // ── Pending MIDI (worker → run()) ─────────────────────────────────────
    // Protected by midiMutex.  Worker appends; run() drains each cycle.
    std::mutex               midiMutex;
    std::vector<PendingNote> pendingMidi;
    std::set<uint8_t>        activeSet; // pitches currently note-on'd
};

// ── Worker thread ─────────────────────────────────────────────────────────────

static void runWorker(NeuralNotePlugin* self)
{
    std::vector<float> snap;

    while (true) {
        // Wait for a snapshot to process
        {
            std::unique_lock<std::mutex> lk(self->workerMutex);
            self->workerCv.wait(lk, [self]{
                return self->workerHasWork || self->workerQuit;
            });
            if (self->workerQuit) break;
            snap = std::move(self->workerSnapshot);
            // Clear the flag *before* releasing the lock so run() can
            // queue the next snapshot while this inference is still running.
            self->workerHasWork = false;
        }

        // Inference — no lock held; this is the slow part (~380 ms on Pi 4)
        self->basicPitch->setParameters(
            self->noteSensitivity.load(std::memory_order_relaxed),
            self->splitSensitivity.load(std::memory_order_relaxed),
            self->minNoteLengthMs.load(std::memory_order_relaxed));
        self->basicPitch->transcribeToMIDI(snap.data(), static_cast<int>(snap.size()));

        // Build pitch set from inference result
        std::set<uint8_t>          newSet;
        std::map<uint8_t, uint8_t> velMap;
        for (const auto& ev : self->basicPitch->getNoteEvents()) {
            const auto p = static_cast<uint8_t>(std::clamp(ev.pitch, 0, 127));
            const auto v = static_cast<uint8_t>(
                std::clamp(static_cast<int>(ev.amplitude * 127.0), 1, 127));
            newSet.insert(p);
            auto it = velMap.find(p);
            if (it == velMap.end() || v > it->second)
                velMap[p] = v;
        }

        // Diff against activeSet → note-on / note-off events
        {
            std::lock_guard<std::mutex> lk(self->midiMutex);
            for (auto p : newSet)
                if (!self->activeSet.count(p))
                    self->pendingMidi.push_back({true, p, velMap[p]});
            for (auto p : self->activeSet)
                if (!newSet.count(p))
                    self->pendingMidi.push_back({false, p, 0});
            self->activeSet = std::move(newSet);
        }
    }

    // Shutdown: release every active note
    std::lock_guard<std::mutex> lk(self->midiMutex);
    for (auto p : self->activeSet)
        self->pendingMidi.push_back({false, p, 0});
    self->activeSet.clear();
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

// Resample one host audio block to 22050 Hz and push into the ring buffer.
static void pushToRing(NeuralNotePlugin* self, const float* in, int inLen)
{
    const int rs = self->ringSize;
    if (self->sampleRate == 22050.0) {
        for (int i = 0; i < inLen; ++i) {
            self->ringBuf[self->ringHead] = in[i];
            self->ringHead = (self->ringHead + 1) % rs;
            if (self->ringFilled < rs) ++self->ringFilled;
        }
        return;
    }
    const double ratio  = 22050.0 / self->sampleRate;
    const int    outLen = static_cast<int>(inLen * ratio);
    for (int i = 0; i < outLen; ++i) {
        const double srcPos = i / ratio;
        const int    s0     = static_cast<int>(srcPos);
        const double frac   = srcPos - s0;
        const int    s1     = std::min(s0 + 1, inLen - 1);
        self->ringBuf[self->ringHead] =
            static_cast<float>((1.0 - frac) * in[s0] + frac * in[s1]);
        self->ringHead = (self->ringHead + 1) % rs;
        if (self->ringFilled < rs) ++self->ringFilled;
    }
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

    self->basicPitch = std::make_unique<BasicPitch>();
    self->basicPitch->setParameters(0.7f, 0.5f, 50.0f);

    memset(self->ringBuf, 0, sizeof(self->ringBuf));

    // Start background inference worker
    self->workerThread = std::thread(runWorker, self);

    lv2_log_note(&self->logger,
                 "NeuralNote Guitar2MIDI: %.0f Hz  [impl: " NEURALNOTE_IMPL_NAME
                 "]  [mode: Fast/Medium/Slow via port 3]\n", rate);
    return static_cast<LV2_Handle>(self);
}

static void connectPort(LV2_Handle instance, uint32_t port, void* data)
{
    NeuralNotePlugin* self = static_cast<NeuralNotePlugin*>(instance);
    switch (static_cast<PortIndex>(port)) {
        case PORT_AUDIO_IN:  self->audioIn   = static_cast<const float*>(data);       break;
        case PORT_MIDI_OUT:  self->midiOut   = static_cast<LV2_Atom_Sequence*>(data); break;
        case PORT_THRESHOLD: self->threshold = static_cast<const float*>(data);       break;
        case PORT_MODE:      self->mode      = static_cast<const float*>(data);       break;
    }
}

static void activate(LV2_Handle instance)
{
    NeuralNotePlugin* self = static_cast<NeuralNotePlugin*>(instance);
    // Reset ring buffer (keep current ringSize / mode)
    self->ringHead   = 0;
    self->ringFilled = 0;
    memset(self->ringBuf, 0, sizeof(self->ringBuf));
    // Reset inference engine
    self->basicPitch->reset();
    // Release all held notes
    {
        std::lock_guard<std::mutex> lk(self->midiMutex);
        for (auto p : self->activeSet)
            self->pendingMidi.push_back({false, p, 0});
        self->activeSet.clear();
    }
}

// ── run() ─────────────────────────────────────────────────────────────────────

static void run(LV2_Handle instance, uint32_t nSamples)
{
    NeuralNotePlugin* self = static_cast<NeuralNotePlugin*>(instance);

    // Set up Atom output sequence
    lv2_atom_forge_set_buffer(&self->forge,
                               reinterpret_cast<uint8_t*>(self->midiOut),
                               self->midiOut->atom.size);
    LV2_Atom_Forge_Frame seqFrame;
    lv2_atom_forge_sequence_head(&self->forge, &seqFrame, 0);

    // Deliver MIDI events produced by the last completed inference
    {
        std::lock_guard<std::mutex> lk(self->midiMutex);
        for (const auto& pn : self->pendingMidi)
            writeMidi(&self->forge, 0, self->uris.midi_MidiEvent,
                      pn.noteOn ? uint8_t(0x90) : uint8_t(0x80),
                      pn.pitch,
                      pn.noteOn ? pn.velocity : uint8_t(0));
        self->pendingMidi.clear();
    }

    // Update sensitivity from control port
    if (self->threshold)
        self->noteSensitivity.store(*self->threshold, std::memory_order_relaxed);

    // Handle mode changes (Fast / Medium / Slow window size).
    // When the mode port changes we reset the ring so the worker immediately
    // starts using the new window length; active notes get note-offs.
    if (self->mode) {
        const int newRingSize = modeToRingSize(*self->mode);
        if (newRingSize != self->ringSize) {
            self->ringSize   = newRingSize;
            self->ringHead   = 0;
            self->ringFilled = 0;
            memset(self->ringBuf, 0, newRingSize * sizeof(float));
            self->basicPitch->reset();
            std::lock_guard<std::mutex> lk(self->midiMutex);
            for (auto p : self->activeSet)
                self->pendingMidi.push_back({false, p, 0});
            self->activeSet.clear();
        }
    }

    // Resample host block to 22050 Hz and push into circular ring buffer
    pushToRing(self, self->audioIn, static_cast<int>(nSamples));

    // Hand the latest ring snapshot to the worker (non-blocking).
    // try_to_lock ensures we never stall the audio thread.
    // workerHasWork is cleared by the worker the moment it takes the snapshot,
    // so run() can queue a fresh snapshot while inference is still running.
    const int rs = self->ringSize;
    if (self->ringFilled >= rs) {
        std::unique_lock<std::mutex> lk(self->workerMutex, std::try_to_lock);
        if (lk.owns_lock() && !self->workerHasWork && !self->workerQuit) {
            self->workerSnapshot.resize(rs);
            for (int i = 0; i < rs; ++i)
                self->workerSnapshot[i] =
                    self->ringBuf[(self->ringHead + i) % rs];
            self->workerHasWork = true;
            lk.unlock();
            self->workerCv.notify_one();
        }
    }

    lv2_atom_forge_pop(&self->forge, &seqFrame);
}

static void deactivate(LV2_Handle /*instance*/) {}

static void cleanup(LV2_Handle instance)
{
    NeuralNotePlugin* self = static_cast<NeuralNotePlugin*>(instance);
    {
        std::lock_guard<std::mutex> lk(self->workerMutex);
        self->workerQuit = true;
    }
    self->workerCv.notify_one();
    if (self->workerThread.joinable())
        self->workerThread.join();
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
