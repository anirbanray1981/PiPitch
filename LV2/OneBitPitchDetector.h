#pragma once
#include <algorithm>
#include <cmath>

// ── OBP voting buffer ─────────────────────────────────────────────────────────
// Tracks one candidate note over a sliding window of OBP detections.
// A note is elected when it wins a supermajority of the last WINDOW cycles.
// Consecutive-run voting buffer.
//
// Requires N_CONSEC consecutive OBP callbacks that all agree on the same note
// before electing it.  Callbacks where OBP returns -1 (not enough zero-crossings
// accumulated yet) are silently ignored — they neither advance nor reset the run.
// A callback that returns a *different* note resets the run to 1.
//
// This fires faster than a window-majority buffer (especially for high-frequency
// notes where OBP produces a result every callback) while being equally or more
// strict: no gaps are tolerated once positive detections begin.
struct OBPVotingBuffer {
    static constexpr int N_CONSEC_BASS = 2;  // E2–A2: long periods, fewer readings needed
    static constexpr int N_CONSEC_MID  = 4;  // A#2–E4: standard
    static constexpr int N_CONSEC_HIGH = 8;  // > E4: more readings to avoid pick chirps
    static constexpr int A2_MIDI       = 45; // E2=40 .. A2=45
    static constexpr int E4_MIDI       = 64;

    int note = -1;
    int run  = 0;

    // Call once per OBP detection cycle. detected == -1 means no pitch found.
    // Returns the elected note number [0..127] after N_CONSEC consecutive matches;
    // -1 otherwise.  High notes (> E4) require more readings to avoid chirps
    // from pick noise harmonics.
    int update(int detected) noexcept {
        if (detected < 0) return -1;  // no result yet — preserve current run
        if (detected == note) {
            const int threshold = (note > E4_MIDI)  ? N_CONSEC_HIGH :
                                  (note <= A2_MIDI) ? N_CONSEC_BASS : N_CONSEC_MID;
            if (++run >= threshold) return note;
        } else {
            note = detected;
            run  = 1;
        }
        return -1;
    }

    void reset() noexcept { note = -1; run = 0; }
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
    // One period per reading — the voting buffer (N_CONSEC=4) handles stability.
    static constexpr int   N_AVG      = 1;

    // Minimum valid periods before a pitch is reported (≤ N_AVG).
    static constexpr int   MIN_VALID  = 1;

    // Schmitt threshold = HYST_RATIO × post-filter amplitude EMA.
    // Near zero (low ratio) all harmonics are in phase with the fundamental,
    // so the crossing time is unbiased by harmonic content → accurate period.
    // Near peak (high ratio) the 2nd harmonic shifts the crossing earlier,
    // causing the apparent period to be ~6-12 samples shorter → measured note
    // comes out 1-2 semitones sharp (e.g. B4/A#4 when A4 is played).
    // 0.25 balances accuracy vs. susceptibility to residual post-filter noise.
    static constexpr float HYST_RATIO = 0.25f;

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

    // Full reset (filter + detection).  Use only when sample rate changes.
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

    // Detection-only reset: preserves filter state so there is no transient
    // when a new onset hits the filter with a sudden step input.
    void resetDetection()
    {
        counter     = 0;
        writeIdx    = 0;
        validCount  = 0;
        schmittHigh = false;
        // filtAmpEMA preserved — keeps threshold calibrated to current signal level
        // filter z-states preserved — avoids transient oscillation on pick attack
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
