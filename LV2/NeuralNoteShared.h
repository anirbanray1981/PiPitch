#pragma once
/**
 * NeuralNoteShared.h — constants, primitives, and pitch-detection pipeline
 * helpers shared between neuralnote_impl.cpp (LV2 plugin) and
 * neuralnote_tune.cpp (JACK tuning tool).
 *
 * Keep this header free of LV2, JACK, and JUCE dependencies.
 *
 * MPM (McLeod Pitch Method) gating
 * ─────────────────────────────────
 * Define NEURALNOTE_ENABLE_MPM before including this header to compile in
 * McLeod calls inside armOrExpireOBP, resetOBPOnGate, and runOBPHPS.
 * neuralnote_impl.cpp defines it only when __ARM_FEATURE_DOTPROD is present
 * (Pi 5 / armv82 build).  neuralnote_tune.cpp always defines it.
 */

#include <algorithm>
#include <atomic>
#include <cstring>
#include <cstdint>
#include <semaphore.h>
#include <vector>

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
static constexpr float ONSET_BLANK_MS = 50.0f;  // re-trigger suppression window

// ── MIDI output queue ─────────────────────────────────────────────────────────

static constexpr int MIDI_QUEUE_CAP = 64; // events between audio callbacks; 64 is ample

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

// ── OBP onset arm / expire ────────────────────────────────────────────────────
//
// Call once per audio callback per range (before the OBP active block).
// On onset: arms OBP window, clears HPS register and blacklist, resets MPM.
// On countdown: decrements obdWindowRemain; disarms when it reaches zero.

template<typename RangeT>
static void armOrExpireOBP(RangeT& r, float sampleRate, int nSamples, bool onsetFired) noexcept
{
    if (onsetFired) {
        r.obdBlacklistNote.store(-1, std::memory_order_relaxed);
        r.obpHpsBits      = 0;
        r.obdOnsetActive  = true;
        r.obdWindowRemain = static_cast<int>(sampleRate * 0.1f);
        r.obdVoting.reset();
        r.obd.reset();
#ifdef NEURALNOTE_ENABLE_MPM
        r.mpm.reset();
#endif
    } else if (r.obdWindowRemain > 0) {
        r.obdWindowRemain -= nSamples;
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
    r.obpHpsBits      = 0;
    r.obdOnsetActive  = false;
    r.obdWindowRemain = 0;
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

        // Accumulate all in-bitmap detections for the HPS register
        if (op >= NOTE_BASE && op < NOTE_BASE + NOTE_COUNT)
            r.obpHpsBits |= (1ULL << (op - NOTE_BASE));

        const int voted = r.obdVoting.update(
            (op >= r.cfg.midiLow && op <= r.cfg.midiHigh && op <= OBP_NOTE_CAP) ? op : -1);

        if (voted != -1) {
            // ── Layer 1: Cross-range harmonic suppression ─────────────────────
            bool isHarmonic = false;
            for (const auto& other : allRanges) {
                if (other.get() == &r) continue;
                const int op2 = other->provNote.load(std::memory_order_relaxed);
                if (op2 != -1) {
                    const int diff = voted - op2;
                    if (diff == 12 || diff == 24) { isHarmonic = true; break; }
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
    RangeT& r, bool onsetFired,
    const float* ringData, double provOnMs, sem_t& workerSem)
{
    const int rs = r.ringSize;
    if (r.snapChan.ready.load(std::memory_order_acquire)) return;
    if (r.ringFilled < rs) return;
    if (!onsetFired && r.freshSamples < r.minFreshSamples) return;

    const int tail = r.ringHead;
    const int p1   = rs - tail;
    std::memcpy(r.snapChan.data.data(),       ringData + tail, p1 * sizeof(float));
    if (p1 < rs)
        std::memcpy(r.snapChan.data.data() + p1, ringData,     (rs - p1) * sizeof(float));
    r.snapChan.snapshotSize       = rs;
    r.snapChan.provNoteAtDispatch = r.provNote.load(std::memory_order_relaxed);
    r.snapChan.provOnMs           = provOnMs;
    r.snapChan.ready.store(true, std::memory_order_release);
    r.freshSamples = 0;
    sem_post(&workerSem);
}
