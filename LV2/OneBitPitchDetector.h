#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>

// ── Platform-optimized 64-bit popcount ───────────────────────────────────────
#if defined(__aarch64__)
#  include <arm_neon.h>
static inline int obp_popcount64(uint64_t x) noexcept {
    const uint8x8_t bytes = vcnt_u8(vcreate_u8(x));
    return static_cast<int>(vaddv_u8(bytes));
}
#elif defined(__GNUC__) || defined(__clang__)
static inline int obp_popcount64(uint64_t x) noexcept {
    return __builtin_popcountll(x);
}
#else
static inline int obp_popcount64(uint64_t x) noexcept {
    x -= (x >> 1) & UINT64_C(0x5555555555555555);
    x  = (x & UINT64_C(0x3333333333333333)) + ((x >> 2) & UINT64_C(0x3333333333333333));
    x  = (x + (x >> 4)) & UINT64_C(0x0f0f0f0f0f0f0f0f);
    return static_cast<int>((x * UINT64_C(0x0101010101010101)) >> 56);
}
#endif

// ── OBP voting buffer ─────────────────────────────────────────────────────────
// Tracks one candidate note over a sliding window of OBP detections.
// A note is elected when it wins a supermajority of the last WINDOW cycles.
struct OBPVotingBuffer {
    static constexpr int      WINDOW = 12;
    static constexpr float    MAJORITY = 0.70f;
    static constexpr int      THRESH   = static_cast<int>(WINDOW * MAJORITY + 0.999f); // 9
    static constexpr uint64_t MASK     = (UINT64_C(1) << WINDOW) - UINT64_C(1);

    int      note = -1;
    uint64_t bits = 0;

    // Call once per OBP detection cycle. detected == -1 means no pitch found.
    // Returns the elected note number [0..127] when THRESH votes are set; -1 otherwise.
    int update(int detected) noexcept {
        bits = (bits << 1) & MASK;
        if (detected >= 0) {
            if (detected != note) { note = detected; bits = 1; }
            else                  { bits |= 1; }
        }
        if (note >= 0 && obp_popcount64(bits) >= THRESH) return note;
        return -1;
    }

    void reset() noexcept { note = -1; bits = 0; }
};

/**
 * OneBitPitchDetector — monophonic pitch detector.
 *
 * Combines three stages to accurately detect the fundamental frequency of a
 * guitar note in the presence of strong harmonics:
 *
 *   1. 4th-order Butterworth lowpass  (two cascaded biquads)
 *      Rejects harmonics above ~1.5× the range's maximum frequency before
 *      zero-crossing detection. -24 dB/oct rolloff minimises harmonic leakage
 *      into adjacent octave ranges.
 *
 *   2. Adaptive Schmitt trigger
 *      Counts a crossing only when the signal transitions from below -T to
 *      above +T, where T = HYST_RATIO × block RMS.  Small harmonic-induced
 *      oscillations near zero are ignored; only the main fundamental swing
 *      clears both thresholds.
 *
 *   3. Period averaging (N_AVG consecutive periods)
 *      Reduces remaining measurement jitter before converting to MIDI.
 *
 * Usage:
 *   Call setLowpass(rangeMaxFreq * 1.5f, sampleRate) once after construction.
 *   Call reset() on silence / plugin activate.
 *   Call process() every audio callback — stateful across consecutive calls.
 *
 * Complexity: O(N) per block — a fixed handful of float ops per sample.
 * No heap allocation.  Safe to call from a real-time audio thread.
 */
struct OneBitPitchDetector {
    // Number of consecutive periods to average before reporting.
    // Kept small (2) since the voting buffer in neuralnote_tune handles stability.
    static constexpr int   N_AVG      = 2;

    // Minimum valid periods before a pitch is reported (≤ N_AVG).
    static constexpr int   MIN_VALID  = 2;

    // Schmitt threshold = HYST_RATIO × block RMS.
    // Higher values tolerate stronger harmonics at the cost of needing a
    // slightly louder signal.  0.5 works well for typical guitar levels.
    static constexpr float HYST_RATIO = 0.5f;

    // ── 4th-order Butterworth lowpass: two cascaded biquads ───────────────
    // Stage 1 Q = 0.5412, Stage 2 Q = 1.3066 (Butterworth pole pairs).
    float b0s1=1,b1s1=0,b2s1=0, a1s1=0,a2s1=0;  // stage-1 coefficients
    float b0s2=1,b1s2=0,b2s2=0, a1s2=0,a2s2=0;  // stage-2 coefficients
    float z1s1=0, z2s1=0;   // stage-1 state (transposed direct form II)
    float z1s2=0, z2s2=0;   // stage-2 state

    // ── Schmitt trigger state ─────────────────────────────────────────────
    bool  schmittHigh   = false;   // true while signal is above +T
    float filtAmpEMA    = 0.001f;  // fast EMA of |filtered sample| — tracks post-filter amplitude
    // EMA alpha per sample: ~0.1 → time constant ≈ 10 samples ≈ 0.2 ms @ 48 kHz
    static constexpr float FILT_EMA_ALPHA = 0.1f;

    // ── Period buffer ─────────────────────────────────────────────────────
    int counter    = 0;
    int buf[N_AVG] = {};
    int writeIdx   = 0;
    int validCount = 0;

    // ── Configure the lowpass (call once when sample rate is known) ───────
    // fc  — cutoff frequency in Hz (use rangeMaxFreq × ~1.5)
    // fs  — sample rate in Hz
    void setLowpass(float fc, float fs)
    {
        if (fc <= 0.0f || fc >= fs * 0.49f) return;  // out of range: leave as bypass
        calcBiquad(fc, fs, 0.5412f, b0s1, b1s1, b2s1, a1s1, a2s1);
        calcBiquad(fc, fs, 1.3066f, b0s2, b1s2, b2s2, a1s2, a2s2);
        z1s1 = z2s1 = z1s2 = z2s2 = 0.0f;
    }

    void reset()
    {
        counter     = 0;
        writeIdx    = 0;
        validCount  = 0;
        schmittHigh = false;
        filtAmpEMA  = 0.001f;
        z1s1 = z2s1 = z1s2 = z2s2 = 0.0f;
        std::fill(buf, buf + N_AVG, 0);
    }

    /**
     * Process one block of audio.  Stateful across consecutive calls.
     * Returns MIDI note [0..127] or -1 if no clear pitch detected yet.
     */
    int process(const float* samples, int nSamples, float sampleRate)
    {
        const int minPer = static_cast<int>(sampleRate / 3000.0f);  // ~16 @ 48 kHz
        const int maxPer = static_cast<int>(sampleRate / 60.0f);    // 800 @ 48 kHz

        // Schmitt threshold is derived from the POST-FILTER amplitude (filtAmpEMA),
        // not raw-block RMS.  Using pre-filter RMS would set the threshold too high
        // when the lowpass strongly attenuates the signal (e.g. A4 through the C3-B3
        // cutoff at 371 Hz), causing the Schmitt trigger to skip cycles and measure
        // sub-harmonics (e.g. D3 instead of A4).
        for (int i = 0; i < nSamples; ++i) {
            // Stage 1 biquad
            const float x1 = samples[i];
            const float y1 = b0s1 * x1 + z1s1;
            z1s1 = b1s1 * x1 - a1s1 * y1 + z2s1;
            z2s1 = b2s1 * x1 - a2s1 * y1;

            // Stage 2 biquad (input is stage-1 output)
            const float y2 = b0s2 * y1 + z1s2;
            z1s2 = b1s2 * y1 - a1s2 * y2 + z2s2;
            z2s2 = b2s2 * y1 - a2s2 * y2;
            // y2 = lowpass-filtered sample

            // Track filtered amplitude with a fast EMA (time constant ~0.2 ms)
            filtAmpEMA = FILT_EMA_ALPHA * std::abs(y2) + (1.0f - FILT_EMA_ALPHA) * filtAmpEMA;
            const float hiT =  filtAmpEMA * HYST_RATIO;
            const float loT = -hiT;

            ++counter;

            // Schmitt trigger: count a rising edge only when transitioning
            // from the LOW state (below loT) to the HIGH state (above hiT).
            if (!schmittHigh) {
                if (y2 >= hiT) {
                    // Rising edge: record period since last rising edge.
                    if (counter >= minPer && counter <= maxPer) {
                        buf[writeIdx % N_AVG] = counter;
                        ++writeIdx;
                        if (validCount < N_AVG) ++validCount;
                    }
                    // Out-of-range period: skip (don't reset — single bad
                    // crossing may be a transient; preserve valid history).
                    counter     = 0;
                    schmittHigh = true;
                }
            } else {
                if (y2 <= loT)
                    schmittHigh = false;  // fell below LOW — ready for next rising edge
            }
        }

        if (validCount < MIN_VALID) return -1;

        float sum = 0.0f;
        for (int i = 0; i < validCount; ++i) sum += static_cast<float>(buf[i]);
        const float avgPeriod = sum / static_cast<float>(validCount);

        const float freq  = sampleRate / avgPeriod;
        if (freq < 60.0f || freq > 3000.0f) return -1;

        const float midiF = 69.0f + 12.0f * std::log2f(freq / 440.0f);
        return std::clamp(static_cast<int>(std::round(midiF)), 0, 127);
    }

private:
    // Compute biquad lowpass coefficients for given fc, fs, Q
    // (transposed direct form II, standard Butterworth section).
    static void calcBiquad(float fc, float fs, float Q,
                           float& b0, float& b1, float& b2,
                           float& a1, float& a2)
    {
        const float K    = std::tan(static_cast<float>(M_PI) * fc / fs);
        const float K2   = K * K;
        const float norm = 1.0f / (1.0f + K / Q + K2);
        b0 = K2 * norm;
        b1 = 2.0f * b0;
        b2 = b0;
        a1 = 2.0f * (K2 - 1.0f) * norm;
        a2 = (1.0f - K / Q + K2) * norm;
    }
};
