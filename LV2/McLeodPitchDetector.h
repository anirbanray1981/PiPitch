#pragma once

#include <fftw3.h>
#include <algorithm>
#include <cmath>
#include <cstring>

/**
 * McLeodPitchDetector — monophonic pitch detector using the McLeod Pitch Method (MPM).
 *
 * Algorithm (McLeod & Wyvill, 2005):
 *
 *   1. Accumulate raw audio into a lock-free circular buffer.
 *
 *   2. When analyze() is called, linearise the buffer and compute the biased
 *      autocorrelation via the Wiener–Khinchin theorem:
 *        - Zero-pad x[0..N-1] to length M (next power of 2 ≥ 2N-1)
 *        - Forward r2c FFT  →  X[k]
 *        - Power spectrum   →  A[k] = |X[k]|² / M  (real, imaginary = 0)
 *        - Inverse c2r FFT  →  r[τ]  (linear autocorrelation)
 *      Complexity: O(M log M) — roughly 40× faster than the O(N²) direct sum
 *      for N = 2048 (≈ 40 ms of audio at 48 kHz).
 *
 *   3. Convert autocorrelation to the Normalised Square Difference Function
 *      (NSDF) using a running update for the denominator m'[τ]:
 *        NSDF[τ] = 2 r[τ] / m'[τ]
 *        m'[0]   = 2 Σ x[j]²
 *        m'[τ+1] = m'[τ] − x[τ]² − x[N−1−τ]²
 *      The denominator update is O(N) total — no extra FFT required.
 *
 *   4. Two-pass peak picking:
 *        Pass 1 — find the global NSDF maximum `peak_max`.
 *        Pass 2 — return the FIRST (smallest-lag = highest-frequency) local
 *                 maximum whose value ≥ MPM_K × peak_max.
 *      This reliably selects the fundamental period over sub-harmonics.
 *
 *   5. Parabolic interpolation on the three NSDF samples surrounding the chosen
 *      peak — yields sub-sample lag accuracy, equivalent to "strobe-tuner"
 *      precision without needing a larger buffer.
 *
 *   6. Convert interpolated lag τ* → frequency → MIDI note number (nearest
 *      semitone, constrained to [midiLow, midiHigh]).
 *
 * Usage:
 *   Call init(sampleRate, midiLow, midiHigh) once after sample rate is known.
 *   Call reset() on silence / onset arm.
 *   Call push(samples, n) every audio callback to accumulate native-SR audio.
 *   Call analyze(sr, midiLow, midiHigh) when a pitch result is needed — this
 *   runs the FFT and NSDF computation and is safe to call from the JACK RT
 *   callback provided init() used FFTW_ESTIMATE (no memory allocation at
 *   execute time).
 *
 * Complexity per analyze():
 *   ~2 × FFT(4096) + O(2048) scalar ops ≈ 0.2 ms on Cortex-A76 @ 2.4 GHz.
 *
 * No heap allocation after init().  Not thread-safe; all methods must be
 * called from one thread (the JACK process callback).
 */
struct McLeodPitchDetector {

    // ── Tuneable constants ──────────────────────────────────────────────────────
    // Maximum analysis window in native-SR samples.
    // 2048 @ 48 kHz = 42.7 ms — covers ≥ 3 periods of E2 (82.4 Hz).
    static constexpr int   MPM_BUFSIZE = 2048;

    // FFT size: next power of 2 ≥ 2*MPM_BUFSIZE − 1 = 4095.  Larger = less
    // circular-wrap contamination; 4096 gives exact linear autocorrelation for
    // windows up to 2048 samples.
    static constexpr int   MPM_FFTSIZE = 4096;

    // NSDF key-maximum threshold K (McLeod 2005).
    // First local maximum ≥ K × global_max is selected as the fundamental period.
    // 0.86 is the canonical value; raise to 0.90 to reject weaker sub-harmonics.
    static constexpr float MPM_K = 0.86f;

    // ── Circular audio accumulation buffer ────────────────────────────────────
    float circBuf[MPM_BUFSIZE] = {};
    int   circHead = 0;   // index of next write position
    int   circFill = 0;   // number of valid samples (≤ MPM_BUFSIZE)

    // Minimum number of samples needed before analysis is meaningful:
    // ≥ 2 full periods of the range's lowest frequency.  Set by init().
    int   minFill  = MPM_BUFSIZE;

    // ── Pre-allocated analysis scratch buffers (no heap in analyze()) ─────────
    float linBuf[MPM_BUFSIZE] = {};   // linearised circular window for FFT input
    float nsdfBuf[MPM_BUFSIZE / 2] = {};  // NSDF values for lags [minLag..maxLag]

    // ── FFTW resources (allocated by init(), freed by destructor) ─────────────
    float*         fftwIn  = nullptr;   // [MPM_FFTSIZE] real FFT input (zero-padded)
    fftwf_complex* fftwOut = nullptr;   // [MPM_FFTSIZE/2 + 1] r2c complex output
    float*         fftwAC  = nullptr;   // [MPM_FFTSIZE] c2r real output (autocorrelation)
    fftwf_plan     planFwd = nullptr;   // r2c forward plan
    fftwf_plan     planInv = nullptr;   // c2r inverse plan

    // ── Frequency limits set by init() ────────────────────────────────────────
    float freqLow  = 60.0f;
    float freqHigh = 3000.0f;

    // ── Lifecycle ──────────────────────────────────────────────────────────────

    McLeodPitchDetector()                                       = default;
    McLeodPitchDetector(const McLeodPitchDetector&)             = delete;
    McLeodPitchDetector& operator=(const McLeodPitchDetector&)  = delete;
    McLeodPitchDetector(McLeodPitchDetector&&)                  = delete;
    McLeodPitchDetector& operator=(McLeodPitchDetector&&)       = delete;

    /**
     * Allocate FFTW plans and set per-range frequency bounds.
     * Must be called once — after the audio sample rate is known — and NOT from
     * the RT callback (FFTW planning is not RT-safe, but execute is).
     */
    void init(float sampleRate, int midiLow, int midiHigh)
    {
        freqLow  = midiToFreq(midiLow);
        freqHigh = midiToFreq(midiHigh);

        // Two full periods of the lowest note, rounded up to a multiple of 64.
        const int maxLag = static_cast<int>(sampleRate / freqLow) + 4;
        const int needed = std::min(maxLag * 2, MPM_BUFSIZE);
        minFill = ((needed + 63) & ~63);  // round up to next 64-sample boundary
        if (minFill < 512) minFill = 512; // floor: MPM unreliable below ~10ms
        if (minFill > MPM_BUFSIZE) minFill = MPM_BUFSIZE;

        // Allocate FFTW-aligned memory
        fftwIn  = fftwf_alloc_real(MPM_FFTSIZE);
        fftwOut = fftwf_alloc_complex(MPM_FFTSIZE / 2 + 1);
        fftwAC  = fftwf_alloc_real(MPM_FFTSIZE);

        // FFTW_ESTIMATE: heuristic plan — no benchmark measurements.
        // fftwf_execute() with FFTW_ESTIMATE plans does NOT allocate memory and
        // is safe to call from a POSIX real-time thread.
        planFwd = fftwf_plan_dft_r2c_1d(MPM_FFTSIZE, fftwIn,  fftwOut, FFTW_ESTIMATE);
        planInv = fftwf_plan_dft_c2r_1d(MPM_FFTSIZE, fftwOut, fftwAC,  FFTW_ESTIMATE);

        // Warm-up: execute once so the FFTW codepath is loaded into I-cache
        // before the first real-time invocation.
        std::memset(fftwIn, 0, MPM_FFTSIZE * sizeof(float));
        fftwf_execute(planFwd);
        fftwf_execute(planInv);
    }

    ~McLeodPitchDetector() noexcept
    {
        if (planFwd) fftwf_destroy_plan(planFwd);
        if (planInv) fftwf_destroy_plan(planInv);
        if (fftwIn)  fftwf_free(fftwIn);
        if (fftwOut) fftwf_free(fftwOut);
        if (fftwAC)  fftwf_free(fftwAC);
    }

    /**
     * Flush the accumulation buffer.  Call on silence or when arming a new onset.
     */
    void reset() noexcept
    {
        circHead = 0;
        circFill = 0;
        std::memset(circBuf, 0, sizeof(circBuf));
    }

    /**
     * Append n samples to the circular accumulation buffer.  O(n), no allocation.
     * Safe to call from the JACK RT callback.
     */
    void push(const float* samples, int n) noexcept
    {
        for (int i = 0; i < n; ++i) {
            circBuf[circHead] = samples[i];
            circHead = (circHead + 1) % MPM_BUFSIZE;
            if (circFill < MPM_BUFSIZE) ++circFill;
        }
    }

    /**
     * Run MPM on the current buffer contents.
     *
     * Returns MIDI note [0..127] whose frequency is within [midiLow, midiHigh],
     * or -1 if there is insufficient data or no clear fundamental.
     *
     * RT-safe: calls fftwf_execute() which does not allocate with FFTW_ESTIMATE.
     * Execution time: ≈ 0.15–0.25 ms on Cortex-A76 @ 2.4 GHz (4096-pt FFT × 2).
     */
    int analyze(float sampleRate, int midiLow, int midiHigh) noexcept
    {
        if (circFill < minFill || !planFwd) return -1;

        const int N = circFill;  // actual analysis window length (≤ MPM_BUFSIZE)

        // ── Step 1: Linearise circular buffer → linBuf[0..N-1] ───────────────
        const int start = (circHead - N + MPM_BUFSIZE) % MPM_BUFSIZE;
        if (start + N <= MPM_BUFSIZE) {
            std::memcpy(linBuf, circBuf + start, N * sizeof(float));
        } else {
            const int p1 = MPM_BUFSIZE - start;
            std::memcpy(linBuf,      circBuf + start, p1 * sizeof(float));
            std::memcpy(linBuf + p1, circBuf,         (N - p1) * sizeof(float));
        }

        // ── Step 2: FFT-based autocorrelation (Wiener–Khinchin) ──────────────
        // Copy linBuf into the zero-padded FFTW input buffer.
        std::memcpy(fftwIn, linBuf, N * sizeof(float));
        std::memset(fftwIn + N, 0, (MPM_FFTSIZE - N) * sizeof(float));

        fftwf_execute(planFwd);  // linBuf (zero-padded) → X[k]  (r2c)

        // Power spectrum: A[k] = |X[k]|² / M.
        // Scaling by 1/M here means the IFFT output is the linear autocorrelation
        // directly (FFTW's c2r IFFT does not divide by M internally).
        const float scale = 1.0f / static_cast<float>(MPM_FFTSIZE);
        for (int k = 0; k <= MPM_FFTSIZE / 2; ++k) {
            const float re = fftwOut[k][0], im = fftwOut[k][1];
            fftwOut[k][0] = (re * re + im * im) * scale;
            fftwOut[k][1] = 0.0f;  // real-valued: Hermitian-symmetric IFFT
        }

        fftwf_execute(planInv);  // A[k] → r[τ]  (c2r; fftwAC[τ] = r_lin[τ])

        // ── Step 3: Lag bounds from the range's frequency limits ──────────────
        const int minLag = std::max(2,
            static_cast<int>(sampleRate / (midiToFreq(midiHigh) * 1.05f)));
        const int maxLag = std::min(N / 2 - 1,
            static_cast<int>(sampleRate / (midiToFreq(midiLow)  * 0.95f)) + 2);
        if (maxLag <= minLag) return -1;

        // ── Step 4: NSDF = 2 r[τ] / m'[τ] ───────────────────────────────────
        // m'[0]   = 2 Σ_{j=0}^{N-1} x[j]²
        // m'[τ+1] = m'[τ] − x[τ]² − x[N−1−τ]²
        float mPrime = 0.0f;
        for (int j = 0; j < N; ++j) mPrime += linBuf[j] * linBuf[j];
        mPrime *= 2.0f;

        // Advance m'[0] → m'[minLag] without storing intermediate NSDF values
        for (int tau = 1; tau <= minLag; ++tau)
            mPrime -= linBuf[tau - 1] * linBuf[tau - 1]
                    + linBuf[N - tau] * linBuf[N - tau];

        // Compute NSDF for the lags we care about, storing in nsdfBuf[0..span-1].
        const int span = maxLag - minLag + 1;
        for (int i = 0; i < span; ++i) {
            const int tau = minLag + i;
            nsdfBuf[i] = (mPrime > 1e-10f) ? 2.0f * fftwAC[tau] / mPrime : 0.0f;
            if (i < span - 1)
                mPrime -= linBuf[tau] * linBuf[tau]
                        + linBuf[N - 1 - tau] * linBuf[N - 1 - tau];
        }

        // ── Step 5: Two-pass peak picking ─────────────────────────────────────
        // Pass 1 — global maximum of all local peaks in [minLag, maxLag].
        float peakMax = 0.0f;
        for (int i = 1; i < span - 1; ++i) {
            if (nsdfBuf[i] > nsdfBuf[i - 1] && nsdfBuf[i] >= nsdfBuf[i + 1])
                if (nsdfBuf[i] > peakMax) peakMax = nsdfBuf[i];
        }
        if (peakMax < 0.1f) return -1;  // no significant periodicity

        // Pass 2 — first (smallest lag = highest frequency = fundamental) local
        // maximum whose value ≥ MPM_K × peakMax.
        const float threshold = MPM_K * peakMax;
        int bestIdx = -1;
        for (int i = 1; i < span - 1; ++i) {
            if (nsdfBuf[i] > nsdfBuf[i - 1] && nsdfBuf[i] >= nsdfBuf[i + 1]
                && nsdfBuf[i] >= threshold) {
                bestIdx = i;
                break;
            }
        }
        if (bestIdx < 0) return -1;

        // ── Step 6: Parabolic interpolation ───────────────────────────────────
        // Fit a parabola to (bestIdx−1, y0), (bestIdx, y1), (bestIdx+1, y2) and
        // find the vertex location.  Gives sub-sample lag accuracy.
        const float y0 = nsdfBuf[bestIdx - 1];
        const float y1 = nsdfBuf[bestIdx];
        const float y2 = nsdfBuf[bestIdx + 1];
        float tauInterp = static_cast<float>(minLag + bestIdx);
        const float denom = y0 - 2.0f * y1 + y2;
        if (std::abs(denom) > 1e-10f)
            tauInterp += 0.5f * (y0 - y2) / denom;  // vertex = τ + Δ

        // ── Step 7: Lag → frequency → MIDI ───────────────────────────────────
        if (tauInterp < 1.0f) return -1;
        const float freq  = sampleRate / tauInterp;

        // Sanity check against the range's bounds (5 % tolerance for interpolation)
        const float fLow  = freqLow  * 0.93f;
        const float fHigh = freqHigh * 1.07f;
        if (freq < fLow || freq > fHigh) return -1;

        const float midiF = 69.0f + 12.0f * std::log2f(freq / 440.0f);
        const int   midi  = static_cast<int>(std::round(midiF));
        if (midi < midiLow || midi > midiHigh) return -1;
        return std::clamp(midi, 0, 127);
    }

private:
    static float midiToFreq(int midi) noexcept {
        return 440.0f * std::pow(2.0f, (midi - 69) / 12.0f);
    }
};
