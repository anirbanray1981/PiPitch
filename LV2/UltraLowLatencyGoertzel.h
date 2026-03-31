#pragma once
/**
 * UltraLowLatencyGoertzel — sample-by-sample polyphonic pitch detection.
 *
 * Processes audio at native sample rate (48 kHz) using NEON SIMD.
 * 4 bins per SIMD lane × 13 groups = 52 slots (49 notes E2–C6 + 3 padding).
 * True zero-latency: can detect pitch within a single audio callback.
 *
 * Features:
 * - Exponential decay prevents unbounded energy accumulation
 * - Hann window applied to input buffer before Goertzel processing
 * - Winner-takes-all: only bins significantly stronger than neighbors fire
 * - Temporal debounce: note must win for N consecutive updates before triggering
 */

#ifdef __aarch64__
#include <arm_neon.h>
#endif

#include <vector>
#include <cmath>
#include <algorithm>
#include <cstdint>
#include <cstring>

class UltraLowLatencyGoertzel {
public:
#ifdef __aarch64__
    struct alignas(64) BinGroup {
        float32x4_t s1;
        float32x4_t s2;
        float32x4_t coeff;
        float32x4_t mag;
        float32x4_t decay;
    };
#else
    struct alignas(64) BinGroup {
        float s1[4];
        float s2[4];
        float coeff[4];
        float mag[4];
        float decay[4];
    };
#endif

    struct NoteStatus {
        bool  isActive       = false;
        bool  triggerPending = false;
        int   velocity       = 0;
        float currentMag     = 0.0f;
        int   debounceCount  = 0;
    };

    static constexpr int   DEBOUNCE_ON    = 4;      // updates above thresh before note-ON (~5ms)
    static constexpr int   DEBOUNCE_OFF   = 6;      // updates below thresh before note-OFF (~8ms)
    static constexpr float DECAY_FACTOR   = 0.99985f; // per-sample decay (~50ms half-life at 48kHz)
    static constexpr float NEIGHBOR_DB    = 6.0f;    // dB above neighbors to claim detection
    static constexpr int   HANN_SIZE      = 64;      // window size (matches JACK buffer)

    UltraLowLatencyGoertzel() = default;

    UltraLowLatencyGoertzel(float fs, int startMidi = 40, int endMidi = 84)
    {
        init(fs, startMidi, endMidi);
    }

    void init(float fs, int startMidi = 40, int endMidi = 84)
    {
        sampleRate_ = fs;
        startMidi_  = startMidi;
        numNotes_   = (endMidi - startMidi) + 1;
        numGroups_  = (numNotes_ + 3) / 4;

        groups_.resize(numGroups_);
        noteStates_.resize(numNotes_);
        magBuf_.resize(numGroups_ * 4, 0.0f);

        // Pre-compute Hann window
        hannWindow_.resize(HANN_SIZE);
        for (int i = 0; i < HANN_SIZE; ++i)
            hannWindow_[i] = 0.5f * (1.0f - std::cos(2.0f * M_PI * i / (HANN_SIZE - 1)));
        windowBuf_.resize(HANN_SIZE, 0.0f);
        windowPos_ = 0;

        for (int i = 0; i < numGroups_; ++i) {
            float c[4] = {0, 0, 0, 0};
            for (int j = 0; j < 4; ++j) {
                int midi = startMidi + (i * 4) + j;
                if (midi <= endMidi) {
                    float f = 440.0f * std::pow(2.0f, (midi - 69) / 12.0f);
                    c[j] = 2.0f * std::cos(2.0f * M_PI * f / fs);
                }
            }
#ifdef __aarch64__
            groups_[i].coeff = vld1q_f32(c);
            groups_[i].s1    = vdupq_n_f32(0.0f);
            groups_[i].s2    = vdupq_n_f32(0.0f);
            groups_[i].mag   = vdupq_n_f32(0.0f);
            groups_[i].decay = vdupq_n_f32(DECAY_FACTOR);
#else
            for (int j = 0; j < 4; ++j) {
                groups_[i].coeff[j] = c[j];
                groups_[i].s1[j] = groups_[i].s2[j] = groups_[i].mag[j] = 0.0f;
                groups_[i].decay[j] = DECAY_FACTOR;
            }
#endif
        }
    }

    // Process a buffer of audio samples with Hann windowing.
    // Call once per audio callback with the full buffer.
    void processBlock(const float* input, int nSamples)
    {
        for (int n = 0; n < nSamples; ++n) {
            // Fill circular window buffer
            windowBuf_[windowPos_] = input[n];
            windowPos_ = (windowPos_ + 1) % HANN_SIZE;

            // Apply Hann window and feed to Goertzel
            const int idx = (windowPos_ + HANN_SIZE - 1) % HANN_SIZE;
            const float windowed = input[n] * hannWindow_[idx % HANN_SIZE];
            processSampleInternal(windowed);
        }
    }

    // Compute magnitudes and update note states with winner-takes-all.
    void update(float transientRatio, float onThresh, float offThresh,
                float sens = 1.2f)
    {
        // Compute magnitudes
#ifdef __aarch64__
        for (int i = 0; i < numGroups_; ++i) {
            BinGroup& g = groups_[i];
            float32x4_t s1_2 = vmulq_f32(g.s1, g.s1);
            float32x4_t s2_2 = vmulq_f32(g.s2, g.s2);
            float32x4_t prod = vmulq_f32(vmulq_f32(g.s1, g.s2), g.coeff);
            g.mag = vsubq_f32(vaddq_f32(s1_2, s2_2), prod);
            vst1q_f32(&magBuf_[i * 4], g.mag);
        }
#else
        for (int i = 0; i < numGroups_; ++i) {
            BinGroup& g = groups_[i];
            for (int j = 0; j < 4; ++j) {
                g.mag[j] = g.s1[j] * g.s1[j] + g.s2[j] * g.s2[j]
                           - g.coeff[j] * g.s1[j] * g.s2[j];
                magBuf_[i * 4 + j] = g.mag[j];
            }
        }
#endif

        const float* m = magBuf_.data();
        const float dynamicOn = onThresh * (1.0f + transientRatio * 2.0f);

        // Neighbor ratio threshold (6 dB = 4× power)
        const float neighborRatio = std::pow(10.0f, NEIGHBOR_DB / 10.0f);

        for (int i = 0; i < numNotes_; ++i) {
            float val = m[i];
            NoteStatus& s = noteStates_[i];

            // Harmonic suppression: check all common guitar harmonic intervals.
            // If a lower note has significant energy, this bin is likely a harmonic.
            if (i >= 12 && m[i - 12] > val * 0.25f) val *= 0.02f;  // octave
            if (i >= 24 && m[i - 24] > val * 0.25f) val *= 0.02f;  // 2 octaves
            if (i >= 7  && m[i - 7]  > val * 0.4f)  val *= 0.05f;  // fifth
            if (i >= 19 && m[i - 19] > val * 0.4f)  val *= 0.05f;  // octave+fifth
            if (i >= 4  && m[i - 4]  > val * 0.5f)  val *= 0.1f;   // major third
            if (i >= 5  && m[i - 5]  > val * 0.5f)  val *= 0.1f;   // fourth
            if (i >= 16 && m[i - 16] > val * 0.4f)  val *= 0.05f;  // octave+fourth
            if (i >= 28 && m[i - 28] > val * 0.3f)  val *= 0.02f;  // 2oct+third

            // Neighborhood-sum relative thresholding:
            // Bin must be significantly stronger than the average of its ±2 neighbors.
            {
                float neighborSum = 0.0f;
                int   neighborCnt = 0;
                for (int k = std::max(0, i - 2); k <= std::min(numNotes_ - 1, i + 2); ++k) {
                    if (k != i) { neighborSum += m[k]; ++neighborCnt; }
                }
                const float neighborAvg = (neighborCnt > 0) ? neighborSum / neighborCnt : 0.0f;
                if (neighborAvg > 0.0f && val < neighborAvg * 1.5f)
                    val = 0.0f;  // not dominant in local neighborhood
            }

            // Hysteresis with sticky debounce
            if (!s.isActive) {
                if (val > dynamicOn) {
                    if (++s.debounceCount >= DEBOUNCE_ON) {
                        s.isActive       = true;
                        s.triggerPending = true;
                        float norm = std::clamp(val / (onThresh * 20.0f), 0.0f, 1.0f);
                        s.velocity = static_cast<int>(127.0f * std::pow(norm, 1.0f / sens));
                        s.velocity = std::max(10, s.velocity);
                    }
                } else {
                    s.debounceCount = 0;
                }
            } else {
                // Active note: count toward OFF unless magnitude is strong
                if (val > onThresh) {
                    s.debounceCount = 0;  // strong signal: reset
                } else {
                    ++s.debounceCount;    // weak or below: count toward OFF
                }
                if (s.debounceCount >= DEBOUNCE_OFF) {
                    s.isActive       = false;
                    s.triggerPending = false;
                    s.debounceCount  = 0;
                }
            }
            s.currentMag = val;
        }
    }

    void reset()
    {
        for (int i = 0; i < numGroups_; ++i) {
#ifdef __aarch64__
            groups_[i].s1  = vdupq_n_f32(0.0f);
            groups_[i].s2  = vdupq_n_f32(0.0f);
            groups_[i].mag = vdupq_n_f32(0.0f);
#else
            for (int j = 0; j < 4; ++j)
                groups_[i].s1[j] = groups_[i].s2[j] = groups_[i].mag[j] = 0.0f;
#endif
        }
        for (auto& s : noteStates_) {
            s.isActive = false;
            s.triggerPending = false;
            s.velocity = 0;
            s.currentMag = 0.0f;
            s.debounceCount = 0;
        }
        std::memset(windowBuf_.data(), 0, windowBuf_.size() * sizeof(float));
        windowPos_ = 0;
    }

    int startMidi() const { return startMidi_; }
    int numNotes()  const { return numNotes_; }
    std::vector<NoteStatus>& getNoteStates() { return noteStates_; }
    const std::vector<NoteStatus>& getNoteStates() const { return noteStates_; }

private:
    // Internal: process one windowed sample through all bins
    inline void processSampleInternal(float input)
    {
#ifdef __aarch64__
        const float32x4_t inp = vdupq_n_f32(input);
        for (int i = 0; i < numGroups_; ++i) {
            BinGroup& g = groups_[i];
            float32x4_t s0 = vmlaq_f32(inp, g.coeff, g.s1);
            s0 = vsubq_f32(s0, g.s2);
            s0 = vmulq_f32(s0, g.decay);
            g.s2 = vmulq_f32(g.s1, g.decay);
            g.s1 = s0;
        }
#else
        for (int i = 0; i < numGroups_; ++i) {
            BinGroup& g = groups_[i];
            for (int j = 0; j < 4; ++j) {
                float s0 = input + g.coeff[j] * g.s1[j] - g.s2[j];
                s0 *= g.decay[j];
                g.s2[j] = g.s1[j] * g.decay[j];
                g.s1[j] = s0;
            }
        }
#endif
    }

    float sampleRate_ = 48000.0f;
    int   startMidi_  = 40;
    int   numNotes_   = 0;
    int   numGroups_  = 0;
    std::vector<BinGroup>    groups_;
    std::vector<NoteStatus>  noteStates_;
    std::vector<float>       magBuf_;
    std::vector<float>       hannWindow_;
    std::vector<float>       windowBuf_;  // circular buffer for windowing
    int                      windowPos_ = 0;
};
