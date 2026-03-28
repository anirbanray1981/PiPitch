#pragma once
/**
 * NeuralNoteShared.h — constants, per-range state base, and pitch-detection
 * pipeline helpers shared between neuralnote_impl.cpp (LV2 plugin) and
 * neuralnote_tune.cpp (JACK tuning tool).
 *
 * Keep this header free of LV2 and JACK dependencies.
 *
 * MPM (McLeod Pitch Method) gating
 * ─────────────────────────────────
 * Define NEURALNOTE_ENABLE_MPM before including this header to compile in
 * McLeod calls inside armOrExpireOBP, resetOBPOnGate, runOBPHPS, and
 * RangeStateBase.  neuralnote_impl.cpp defines it before its includes when
 * __ARM_FEATURE_DOTPROD is present (Pi 5 / armv82 build).
 * neuralnote_tune.cpp always defines it.
 */

#include <algorithm>
#include <atomic>
#include <cstring>
#include <cstdint>
#include <memory>
#include <semaphore.h>
#include <vector>

// BinaryData.h must precede any Lib/Model header.
#include "BinaryData.h"
#include "BasicPitch.h"
#include "NoteRangeConfig.h"
#include "OneBitPitchDetector.h"
#ifdef NEURALNOTE_ENABLE_MPM
#  include "McLeodPitchDetector.h"
#endif

// ── Sample rate ───────────────────────────────────────────────────────────────
// BasicPitch always operates at 22050 Hz; all ring buffers use this rate.

static constexpr double PLUGIN_SR = 22050.0;

// ── Guitar MIDI range ─────────────────────────────────────────────────────────
// E2 (MIDI 40) … E6 (MIDI 88) — 49 notes, fits in one uint64_t bitmap.

static constexpr int NOTE_BASE  = 40;
static constexpr int NOTE_COUNT = 49;

static inline void bmSet  (uint64_t& b, int midi) noexcept { b |=  (1ULL << (midi - NOTE_BASE)); }
static inline void bmClear(uint64_t& b, int midi) noexcept { b &= ~(1ULL << (midi - NOTE_BASE)); }
static inline bool bmTest (uint64_t  b, int midi) noexcept { return (b >> (midi - NOTE_BASE)) & 1; }

// ── Ring-buffer limits ────────────────────────────────────────────────────────

static constexpr int RING_MAX        = static_cast<int>(PLUGIN_SR * 2.0); // 44100 samples
static constexpr int MIN_FRESH_FLOOR = static_cast<int>(PLUGIN_SR * 0.025); // ~25 ms

// ── Onset detection ───────────────────────────────────────────────────────────

static constexpr float ONSET_RATIO    = 3.0f;   // RMS must exceed background × this
static constexpr float ONSET_ALPHA    = 0.05f;  // background tracker time constant
static constexpr float ONSET_BLANK_MS = 25.0f;  // re-trigger suppression window

// ── HPF pick detector ─────────────────────────────────────────────────────────
//
// Detects pick attacks via high-frequency attack slope.  A 1st-order IIR HPF
// at ~3 kHz isolates the broadband "snap" of a pick hitting a string.  Two
// envelope EMAs at different speeds track the HPF output:
//   - envFast (~0.3 ms): responds to transients within 1 ms
//   - envSlow (~5 ms):   smooths out harmonic oscillations
// A pick creates a large gap (fast >> slow).  Harmonic beating keeps them close.

struct PickDetector {
    float hpfPrev    = 0.0f;    // x[n-1]
    float hpfOut     = 0.0f;    // y[n-1]
    float hpfAlpha   = 0.0f;    // HPF coefficient

    float envFast    = 1e-6f;   // fast envelope (~0.3 ms)
    float envSlow    = 1e-6f;   // slow envelope (~5 ms)
    float fastAlpha  = 0.0f;    // EMA coefficient for fast
    float slowAlpha  = 0.0f;    // EMA coefficient for slow

    float slopeRatio = 10.0f;   // fast/slow threshold to fire
    int   blankTotal = 0;       // suppression window (samples)
    int   blankRemain = 0;      // remaining suppression samples

    void init(float sampleRate, float cutoffHz = 3000.0f,
              float ratio = 4.0f, float blankMs = 25.0f)
    {
        const float dt = 1.0f / sampleRate;
        const float RC = 1.0f / (2.0f * 3.14159265f * cutoffHz);
        hpfAlpha   = RC / (RC + dt);
        fastAlpha  = 1.0f - std::exp(-1.0f / (sampleRate * 0.0001f));  // ~0.1 ms
        slowAlpha  = 1.0f - std::exp(-1.0f / (sampleRate * 0.02f));   // ~20 ms
        slopeRatio = ratio;
        blankTotal = static_cast<int>(sampleRate * blankMs / 1000.0f);
    }

    // Process a buffer.  Returns the sample index where a pick was detected,
    // or -1 if none.  Also writes the fast/slow ratio at the trigger point.
    int process(const float* audio, int nSamples, float& ratioOut)
    {
        for (int i = 0; i < nSamples; ++i) {
            // 1st-order IIR HPF: y[n] = α·(y[n-1] + x[n] - x[n-1])
            hpfOut  = hpfAlpha * (hpfOut + audio[i] - hpfPrev);
            hpfPrev = audio[i];

            const float absHpf = std::fabs(hpfOut);

            // Dual-EMA envelope tracking
            envFast += fastAlpha * (absHpf - envFast);
            envSlow += slowAlpha * (absHpf - envSlow);

            if (blankRemain > 0) { --blankRemain; continue; }

            if (envSlow > 1e-7f && envFast > envSlow * slopeRatio) {
                blankRemain = blankTotal;
                ratioOut    = envFast / envSlow;
                envSlow     = envFast;  // snap slow to fast; must rebuild gap
                return i;
            }
        }
        return -1;
    }
};

// ── MIDI output queue ─────────────────────────────────────────────────────────

static constexpr int MIDI_QUEUE_CAP = 64; // events between audio callbacks; 64 is ample

// ── Pending MIDI event ────────────────────────────────────────────────────────
// velocity: 0–127.  LV2 consumer casts to uint8_t; synth consumer divides by 127.f.

struct PendingNote { bool noteOn; int pitch; int velocity; };

// ── Lockless SPSC MIDI output queue (worker → audio thread) ──────────────────
// Standard single-producer / single-consumer ring buffer.
// Producer: worker thread.  Consumer: audio/process callback.

struct MidiOutQueue {
    PendingNote      buf[MIDI_QUEUE_CAP];
    std::atomic<int> head{0};
    std::atomic<int> tail{0};

    void push(PendingNote n) {
        const int t    = tail.load(std::memory_order_relaxed);
        const int next = (t + 1) % MIDI_QUEUE_CAP;
        if (next == head.load(std::memory_order_acquire)) return; // drop if full
        buf[t] = n;
        tail.store(next, std::memory_order_release);
    }
    bool pop(PendingNote& out) {
        const int h = head.load(std::memory_order_relaxed);
        if (h == tail.load(std::memory_order_acquire)) return false;
        out = buf[h];
        head.store((h + 1) % MIDI_QUEUE_CAP, std::memory_order_release);
        return true;
    }
};

// ── Window helper ─────────────────────────────────────────────────────────────

static inline int windowMsToRingSize(float ms)
{
    const float clamped = std::clamp(ms, 35.0f, 2000.0f);
    return std::min(static_cast<int>(clamped / 1000.0f * PLUGIN_SR), RING_MAX);
}

// ── Lockless SPSC snapshot channel (audio thread → worker) ───────────────────
//
// Ordering contract:
//   Producer: write data/snapshotSize → ready.store(true, release) → sem_post
//   Consumer: sem_wait → ready.load(acquire) → read data → ready.store(false, release)
//
// The release/acquire on `ready` provides the happens-before edge; sem_post/wait
// additionally ensures the worker sleeps when idle.
//
// provOnMs is only used by neuralnote_tune (latency measurement); the LV2 plugin
// always leaves it at 0.0 and never reads it.

struct SnapshotChannel {
    std::vector<float> data;
    int                snapshotSize       = 0;
    int                provNoteAtDispatch = -1;
    double             provOnMs           = 0.0;
    bool               onsetDispatched    = false;  // true if this snapshot was force-dispatched by an onset
    std::atomic<bool>  ready{false};
    std::atomic<bool>  quit{false};
    sem_t              sem;

    SnapshotChannel()  { data.resize(RING_MAX); sem_init(&sem, 0, 0); }
    ~SnapshotChannel() { sem_destroy(&sem); }
    SnapshotChannel(const SnapshotChannel&)            = delete;
    SnapshotChannel& operator=(const SnapshotChannel&) = delete;
    SnapshotChannel(SnapshotChannel&&)                 = delete;
    SnapshotChannel& operator=(SnapshotChannel&&)      = delete;
};

// ── Per-range state base ──────────────────────────────────────────────────────
//
// Contains all fields shared between RangeState (LV2 plugin) and
// RangeState (JACK tuning tool).  Each file derives its own struct and adds
// file-specific extras (LV2 single-range params / tune hold/logging state).

struct RangeStateBase {
    NoteRange cfg;

    std::unique_ptr<BasicPitch> basicPitch;  // CNN inference engine

    // Ring buffer: 22050-Hz resampled audio (size RING_MAX, logical size ringSize)
    std::vector<float> ring;
    int ringHead        = 0;
    int ringFilled      = 0;
    int freshSamples    = 0;
    int ringSize        = 0;
    int minFreshSamples = 0;  // = max(ringSize/2, MIN_FRESH_FLOOR)

    SnapshotChannel snapChan;  // audio thread → worker (SPSC, lockless)
    MidiOutQueue    midiOut;   // worker → audio thread (SPSC, lockless)

    // Note tracking (worker thread only — no synchronisation needed)
    // Bitmaps over E2 (MIDI 40) … E6 (MIDI 88) — 49 notes in a uint64_t.
    uint64_t activeNotes = 0;
    uint64_t holdNotes   = 0;
    alignas(8) int8_t holdCounts[NOTE_COUNT] = {};

    // Two-phase pitch detection: OBP fires a fast provisional; CNN confirms it.
    OneBitPitchDetector obd;
    std::atomic<int>    provNote{-1};           // set by audio thread, cleared by worker
    OBPVotingBuffer     obdVoting;
    std::atomic<int>    obdBlacklistNote{-1};   // CNN-cancelled note; suppressed next onset
    std::atomic<int>    monoHeldNote{-1};       // mono mode: current held note (-1 = none)
    std::atomic<bool>     hasActiveNotes{false};   // worker has active/held notes; suppresses RMS gate
    std::atomic<uint64_t> activeNotesBits{0};      // mirror of activeNotes bitmap; audio thread re-hit check
    uint64_t            obpHpsBits         = 0;    // HPS accumulator for this onset window
    bool                obdOnsetActive     = false;
    int                 obdWindowRemain    = 0;
    int                 obdPendingNote     = -1;   // OBP voted but MPM not ready; retry next callback
    int                 obdPendingRemain   = 0;    // sample countdown; clears pending when ≤ 0

    // Worker-only: first CNN cycle after a new provisional fires may contain mostly
    // the previous note's audio.  One grace cycle suppresses a premature cancel.
    int                 provLastSeenByCNN  = -1;   // last provisional note the worker processed
    int                 provCancelGrace    = 0;    // cancel-suppression cycles remaining

    // Worker-only: first SwiftF0 cycle(s) after onset contain stale ring audio.
    // Grace suppresses detections: mid-note keeps current note; from-silence
    // suppresses cycle 1 and remembers the stale note; cycle 2 allows if different.
    int                 swiftOnsetGrace    = 0;
    int                 swiftGraceStaleNote = -1;  // note suppressed on grace cycle 1

#ifdef NEURALNOTE_ENABLE_MPM
    McLeodPitchDetector mpm;  // FFT autocorrelation — agrees with OBP before prov fires
#endif

    // Non-copyable / non-moveable (owns SnapshotChannel which owns sem_t)
    RangeStateBase()                               = default;
    RangeStateBase(const RangeStateBase&)          = delete;
    RangeStateBase& operator=(const RangeStateBase&) = delete;
    RangeStateBase(RangeStateBase&&)               = delete;
    RangeStateBase& operator=(RangeStateBase&&)    = delete;
};

// ── Ring write helper ─────────────────────────────────────────────────────────
//
// Append `count` pre-resampled (22050-Hz) samples into the circular ring.
// Does not perform any sample-rate conversion; caller is responsible for that.

template<typename RangeT>
static void pushRingSamples(RangeT& r, const float* samples, int count) noexcept
{
    const int rs = r.ringSize;
    for (int i = 0; i < count; ++i) {
        r.ring[r.ringHead] = samples[i];
        r.ringHead = (r.ringHead + 1) % rs;
        if (r.ringFilled < rs) ++r.ringFilled;
    }
    r.freshSamples += count;
}

// ── OBP onset arm / expire ────────────────────────────────────────────────────
//
// Call once per audio callback per range (before the OBP active block).
// On first onset: arms OBP window, clears HPS register and blacklist, resets MPM.
// On re-trigger onset (already active): extends window only — does NOT reset OBP/voting
//   state, so accumulated readings are preserved.
// On countdown: decrements obdWindowRemain; disarms when it reaches zero.

template<typename RangeT>
static void armOrExpireOBP(RangeT& r, float sampleRate, int nSamples, bool onsetFired) noexcept
{
    if (onsetFired) {
        if (!r.obdOnsetActive) {
            // First onset: full reset
            r.obdBlacklistNote.store(-1, std::memory_order_relaxed);
            r.obpHpsBits         = 0;
            r.obdPendingNote     = -1;
            r.obdPendingRemain   = 0;
            r.obdVoting.reset();
            r.obd.resetDetection();
#ifdef NEURALNOTE_ENABLE_MPM
            r.mpm.reset();
#endif
        }
        // Arm / extend window (both first onset and re-triggers)
        r.obdOnsetActive  = true;
        r.obdWindowRemain = static_cast<int>(sampleRate * 0.1f);  // 100 ms window
    } else if (r.obdWindowRemain > 0) {
        r.obdWindowRemain  -= nSamples;
        if (r.obdWindowRemain <= 0) {
            r.obdWindowRemain = 0;
            r.obdOnsetActive  = false;
        }
    }
}

// ── OBP gate reset ────────────────────────────────────────────────────────────
//
// Call when the block is below the noise gate.
// Clears all OBP / HPS / MPM state so stale data can't bleed into the next note.

template<typename RangeT>
static void resetOBPOnGate(RangeT& r) noexcept
{
    r.obd.reset();
    r.obdVoting.reset();
#ifdef NEURALNOTE_ENABLE_MPM
    r.mpm.reset();
#endif
    r.obpHpsBits       = 0;
    r.obdOnsetActive   = false;
    r.obdWindowRemain  = 0;
    r.obdPendingNote   = -1;
    r.obdPendingRemain = 0;
}

// ── OBP + HPS voting loop ─────────────────────────────────────────────────────
//
// Runs the OneBitPitch sub-block loop (OBP_CHUNK = 16 samples per iteration).
// On each OBP vote applies two suppression layers:
//
//   Layer 1 — Cross-range harmonic suppression:
//             If another range already fired a provisional that is exactly
//             1 or 2 octaves below our vote, we are its harmonic → suppress.
//
//   Layer 2 — Bit-parallel HPS:
//             OR all ranges' obpHpsBits; intersect with ÷2-octave and
//             ÷3-octave (≈19-semitone) shifts.  The lowest surviving bit
//             is the true fundamental.
//
// Preconditions (caller must ensure):
//   - r.obdOnsetActive == true
//   - r.provNote.load() == -1  (no provisional already pending)
//   - r.mpm.push() has been called for this callback (if MPM is enabled)
//
// Returns the HPS-corrected MIDI note if OBP+HPS pass all gates;
//         -1 if no vote was cast, or if the vote was suppressed.
//
// After returning ≥ 0, caller should run r.mpm.analyze() and fire the
// provisional only if MPM agrees (Pi 5), or fire directly (Pi 4 / no MPM).

template<typename RangeT, typename RangesContainer>
static int runOBPHPS(RangeT& r,
                     const float* audioIn, int nSamples, float sampleRate,
                     const RangesContainer& allRanges) noexcept
{
    static constexpr int OBP_CHUNK    = 16;
    static constexpr int OBP_NOTE_CAP = 76;  // E5 — reject OBP provisionals above this

    const float* obpPtr = audioIn;
    int          obpRem = nSamples;

    while (r.obdOnsetActive && obpRem > 0) {
        const int chunk = std::min(obpRem, OBP_CHUNK);
        const int op    = r.obd.process(obpPtr, chunk, sampleRate);

#ifdef OBP_DIAG
        // Diagnostic: log every raw OBP reading for the lowest range
        if (op != -1 && r.cfg.midiHigh <= 47) {
            const bool inRange = (op >= r.cfg.midiLow && op <= r.cfg.midiHigh && op <= OBP_NOTE_CAP);
            std::printf("  OBP-raw: midi=%2d %s  voting: note=%d run=%d/%d  period=%d\n",
                        op, inRange ? "IN " : "OUT",
                        r.obdVoting.note, r.obdVoting.run, OBPVotingBuffer::N_CONSEC,
                        r.obd.buf[0]);
            std::fflush(stdout);
        }
#endif

        // Accumulate all in-bitmap detections for the HPS register
        if (op >= NOTE_BASE && op < NOTE_BASE + NOTE_COUNT)
            r.obpHpsBits |= (1ULL << (op - NOTE_BASE));

        const int voted = r.obdVoting.update(
            (op >= r.cfg.midiLow && op <= r.cfg.midiHigh && op <= OBP_NOTE_CAP) ? op : -1);

        if (voted != -1) {
            // ── Layer 1: Cross-range harmonic suppression ─────────────────────
            // Check both harmonics (voted above prov) and sub-harmonics (voted below).
            bool isHarmonic = false;
            for (const auto& other : allRanges) {
                if (other.get() == &r) continue;
                const int op2 = other->provNote.load(std::memory_order_relaxed);
                if (op2 != -1) {
                    const int absDiff = std::abs(voted - op2);
                    if (absDiff == 12 || absDiff == 24) { isHarmonic = true; break; }
                }
            }

            // ── Layer 2: Bit-parallel HPS ─────────────────────────────────────
            uint64_t allHps = 0;
            for (const auto& other : allRanges) allHps |= other->obpHpsBits;
            const uint64_t hps2    = allHps & (allHps >> 12);
            const uint64_t hps3    = hps2   & (allHps >> 19);
            const uint64_t hpsBest = hps3 ? hps3 : hps2;
            int finalNote = voted;
            if (hpsBest) {
                const int hpsBit  = __builtin_ctzll(hpsBest);
                const int hpsNote = NOTE_BASE + hpsBit;
                if (hpsNote >= r.cfg.midiLow && hpsNote <= r.cfg.midiHigh)
                    finalNote = hpsNote;    // corrected to true fundamental
                else if (hpsNote < r.cfg.midiLow)
                    isHarmonic = true;      // fundamental belongs to a lower range
            }

            r.obdOnsetActive  = false;
            r.obdWindowRemain = 0;
            r.obdVoting.reset();

            if (!isHarmonic) {
                const int bl = r.obdBlacklistNote.load(std::memory_order_relaxed);
                if (finalNote != bl)
                    return finalNote;
            }
            return -1;  // vote cast but suppressed (harmonic or blacklisted)
        }
        obpPtr += chunk;
        obpRem -= chunk;
    }
    return -1;  // no vote this callback
}

// ── Snapshot dispatch ─────────────────────────────────────────────────────────
//
// Linearises the circular ring buffer into the snapshot slot and wakes the
// worker thread via sem_post.  No-op if:
//   - The slot is already occupied (worker hasn't consumed previous snapshot)
//   - The ring hasn't been filled to ringSize yet (still priming)
//   - Not enough fresh audio has arrived AND onsetFired is false
//
// ringData : pointer to the start of the circular buffer (size ≥ ringSize)
// provOnMs : wall-clock ms when the provisional fired; 0.0 if unused (LV2 path)

template<typename RangeT>
static void dispatchSnapshotIfReady(
    RangeT& r, bool onsetFired, double provOnMs, sem_t& workerSem,
    float gateFloor = 0.003f)
{
    const int rs = r.ringSize;
    if (r.snapChan.ready.load(std::memory_order_acquire)) return;
    if (r.ringFilled < rs) return;
    if (!onsetFired && r.freshSamples < r.minFreshSamples) return;

    // RMS gate: skip CNN dispatch if ring energy is negligible AND the worker
    // has no active or held notes that still require note-off events.
    // Catches sympathetic resonance / string buzz that slips past the onset
    // detector but carries no real note energy.
    // Bypassed when hasActiveNotes is true so decaying notes always get their
    // CNN-driven hold countdown and eventual note-off.
    if (!r.hasActiveNotes.load(std::memory_order_acquire)) {
        const float* rd = r.ring.data();
        float sumSq = 0.0f;
        for (int i = 0; i < rs; ++i) sumSq += rd[i] * rd[i];
        if (sumSq < gateFloor * gateFloor * static_cast<float>(rs)) return;
    }

    const float* ringData = r.ring.data();
    const int    tail     = r.ringHead;
    const int    p1       = rs - tail;
    std::memcpy(r.snapChan.data.data(),       ringData + tail, p1 * sizeof(float));
    if (p1 < rs)
        std::memcpy(r.snapChan.data.data() + p1, ringData,     (rs - p1) * sizeof(float));
    r.snapChan.snapshotSize       = rs;
    r.snapChan.provNoteAtDispatch = r.provNote.load(std::memory_order_relaxed);
    r.snapChan.provOnMs           = provOnMs;
    r.snapChan.onsetDispatched    = onsetFired;
    r.snapChan.ready.store(true, std::memory_order_release);
    r.freshSamples = 0;
    sem_post(&workerSem);
}

// ── CNN output → note bitmap ──────────────────────────────────────────────────
//
// Iterates basicPitch::getNoteEvents(), filters by ampFloor and the range's
// midiLow/midiHigh and bitmap bounds, and fills newBits/newVel.
// Velocity values are 1–127 (int8_t); newVel is zeroed before filling.

static void buildNNBits(RangeStateBase& r, float ampFloor,
                        uint64_t& newBits, int8_t* newVel) noexcept
{
    newBits = 0;
    std::memset(newVel, 0, NOTE_COUNT);
    for (const auto& ev : r.basicPitch->getNoteEvents()) {
        if (static_cast<float>(ev.amplitude) < ampFloor) continue;
        const int p = static_cast<int>(ev.pitch);
        if (p < r.cfg.midiLow || p > r.cfg.midiHigh) continue;
        if (p < NOTE_BASE || p >= NOTE_BASE + NOTE_COUNT) continue;
        const int bit = p - NOTE_BASE;
        const int v   = std::clamp(static_cast<int>(ev.amplitude * 127.0), 1, 127);
        if (!(newBits & (1ULL << bit)) || v > newVel[bit]) {
            newBits    |= (1ULL << bit);
            newVel[bit] = static_cast<int8_t>(v);
        }
    }
}

// ── Note ON/OFF/hold state machine ────────────────────────────────────────────
//
// Applies CNN result (newBits/newVel) to per-range note state and queues MIDI
// events into r.midiOut.  prov is the OBP provisional MIDI note (-1 if none).
//
// Logic:
//   • If prov wasn't confirmed → blacklist it (one-onset suppression).
//   • If prov sits in the hold-countdown queue → force-expire it so OFF fires
//     this cycle rather than waiting holdCycles more inferences.
//   • Note-ONs: cancel any outstanding hold for returning notes, then push ON.
//   • Hold decrement: tick all held notes; push OFF for those that expire.
//   • Vanishing notes: start hold (unless cancelled prov) or push OFF directly.
//
// No logging — callers may compare activeNotes before/after if they need logs.

static void applyNotesDiff(RangeStateBase& r, uint64_t newBits,
                           const int8_t* newVel, int prov, bool mono = false,
                           int holdCyclesOverride = -1) noexcept
{
    const int holdCycles = (holdCyclesOverride >= 0) ? holdCyclesOverride
                                                     : r.cfg.holdCycles;
    // Mono mode: reduce CNN result to single highest-velocity note in this range
    if (mono && newBits) {
        int8_t bestVel = 0; int bestBit = -1;
        for (uint64_t tmp = newBits; tmp; tmp &= tmp - 1) {
            const int bit = __builtin_ctzll(tmp);
            if (newVel[bit] > bestVel) { bestVel = newVel[bit]; bestBit = bit; }
        }
        newBits = (bestBit >= 0) ? (1ULL << bestBit) : 0;
    }

    // Blacklist unconfirmed provisional
    if (prov != -1 && !bmTest(newBits, prov))
        r.obdBlacklistNote.store(prov, std::memory_order_release);

    // Force-expire cancelled prov from hold so OFF fires this cycle
    if (prov != -1 && prov >= NOTE_BASE && prov < NOTE_BASE + NOTE_COUNT
        && !bmTest(newBits, prov)) {
        const int provBit = prov - NOTE_BASE;
        if (r.holdNotes & (1ULL << provBit))
            r.holdCounts[provBit] = 1;
    }

    // Returning notes (were in hold, now seen again by CNN).
    // Just cancel the hold — the MIDI note is already playing (no OFF was sent when
    // entering hold).  Do NOT send OFF+ON here: CNN confidence fluctuation on a
    // sustained note would cause spurious stutters every time a cycle is missed.
    // Same-note staccato re-hits are handled at the provisional level by fireProv
    // reading activeNotesBits and sending OFF before ON.
    const uint64_t returning = newBits & r.holdNotes;
    for (uint64_t tmp = returning; tmp; tmp &= tmp - 1)
        r.holdCounts[__builtin_ctzll(tmp)] = 0;
    r.holdNotes &= ~returning;
    // New note-ONs (not already active or returning)
    for (uint64_t tmp = newBits & ~r.activeNotes; tmp; tmp &= tmp - 1) {
        const int bit = __builtin_ctzll(tmp);
        r.midiOut.push({true, NOTE_BASE + bit, newVel[bit]});
        r.activeNotes |= (1ULL << bit);
    }

    // Decrement holds; OFF expired
    for (uint64_t tmp = r.holdNotes; tmp; tmp &= tmp - 1) {
        const int bit = __builtin_ctzll(tmp);
        if (--r.holdCounts[bit] <= 0) {
            r.midiOut.push({false, NOTE_BASE + bit, 0});
            r.activeNotes &= ~(1ULL << bit);
            r.holdNotes   &= ~(1ULL << bit);
            r.holdCounts[bit] = 0;
        }
    }

    // Active notes absent from newBits → start hold or OFF immediately
    for (uint64_t tmp = r.activeNotes & ~newBits & ~r.holdNotes; tmp; tmp &= tmp - 1) {
        const int  bit          = __builtin_ctzll(tmp);
        const bool cancelledProv = (prov != -1 && (NOTE_BASE + bit) == prov
                                    && !bmTest(newBits, prov));
        if (holdCycles > 0 && !cancelledProv) {
            r.holdNotes      |= (1ULL << bit);
            r.holdCounts[bit]  = static_cast<int8_t>(holdCycles);
        } else {
            r.midiOut.push({false, NOTE_BASE + bit, 0});
            r.activeNotes &= ~(1ULL << bit);
        }
    }

    // Mono: track current held note for cross-range provisional kill in audio thread
    if (mono)
        r.monoHeldNote.store(
            r.activeNotes ? (NOTE_BASE + __builtin_ctzll(r.activeNotes)) : -1,
            std::memory_order_release);

    // Publish note state to the audio thread:
    //   hasActiveNotes — bypasses RMS gate so decaying notes still get note-offs.
    //   activeNotesBits — allows fireProv to detect same-note re-hits and send OFF first.
    r.hasActiveNotes.store(r.activeNotes != 0 || r.holdNotes != 0,
                           std::memory_order_release);
    r.activeNotesBits.store(r.activeNotes, std::memory_order_release);
}
