#pragma once
/**
 * UltraLowLatencyGoertzel — sample-by-sample polyphonic pitch detection.
 *
 * Processes audio at native sample rate (48 kHz) using NEON SIMD.
 * 4 bins per SIMD lane × 13 groups = 52 slots (49 notes E2–C6 + 3 padding).
 * True zero-latency: can detect pitch within a single audio callback.
 *
 * Usage (audio thread):
 *   for each sample:  goertzel.processSample(sample);
 *   every N samples:  goertzel.update(transientRatio, onThresh, offThresh);
 *   read results:     goertzel.getNoteStates()
 */

#ifdef __aarch64__
#include <arm_neon.h>
#endif

#include <vector>
#include <cmath>
#include <algorithm>
#include <cstdint>

class UltraLowLatencyGoertzel {
public:
#ifdef __aarch64__
    struct alignas(64) BinGroup {
        float32x4_t s1;
        float32x4_t s2;
        float32x4_t coeff;
        float32x4_t mag;
    };
#else
    // Scalar fallback for non-ARM platforms
    struct alignas(64) BinGroup {
        float s1[4];
        float s2[4];
        float coeff[4];
        float mag[4];
    };
#endif

    struct NoteStatus {
        bool  isActive       = false;
        bool  triggerPending = false;
        int   velocity       = 0;
        float currentMag     = 0.0f;
    };

    UltraLowLatencyGoertzel() = default;

    UltraLowLatencyGoertzel(float fs, int startMidi = 40, int endMidi = 84)
        : sampleRate_(fs), startMidi_(startMidi)
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
#else
            for (int j = 0; j < 4; ++j) {
                groups_[i].coeff[j] = c[j];
                groups_[i].s1[j] = groups_[i].s2[j] = groups_[i].mag[j] = 0.0f;
            }
#endif
        }
    }

    // Process one audio sample through all Goertzel bins.
    // Call this for every sample at native sample rate (e.g. 48 kHz).
    inline void processSample(float input)
    {
#ifdef __aarch64__
        const float32x4_t inp = vdupq_n_f32(input);
        for (int i = 0; i < numGroups_; ++i) {
            BinGroup& g = groups_[i];
            // Goertzel: s0 = input + coeff * s1 - s2
            float32x4_t s0 = vmlaq_f32(inp, g.coeff, g.s1);
            s0 = vsubq_f32(s0, g.s2);
            g.s2 = g.s1;
            g.s1 = s0;
        }
#else
        for (int i = 0; i < numGroups_; ++i) {
            BinGroup& g = groups_[i];
            for (int j = 0; j < 4; ++j) {
                float s0 = input + g.coeff[j] * g.s1[j] - g.s2[j];
                g.s2[j] = g.s1[j];
                g.s1[j] = s0;
            }
        }
#endif
    }

    // Compute magnitudes and update note states.
    // Call periodically (e.g. once per audio callback or on onset).
    // transientRatio: PickDetector's fast/slow ratio (higher = louder attack)
    // onThresh:  magnitude threshold to trigger note-ON
    // offThresh: magnitude threshold to trigger note-OFF (hysteresis)
    // sens:      velocity curve exponent (1.0 = linear, > 1.0 = compressed)
    void update(float transientRatio, float onThresh, float offThresh,
                float sens = 1.2f)
    {
        // Compute magnitudes for all bins
#ifdef __aarch64__
        for (int i = 0; i < numGroups_; ++i) {
            BinGroup& g = groups_[i];
            float32x4_t s1_2 = vmulq_f32(g.s1, g.s1);
            float32x4_t s2_2 = vmulq_f32(g.s2, g.s2);
            float32x4_t prod = vmulq_f32(vmulq_f32(g.s1, g.s2), g.coeff);
            g.mag = vsubq_f32(vaddq_f32(s1_2, s2_2), prod);
            // Store to flat buffer for scalar note logic
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

        for (int i = 0; i < numNotes_; ++i) {
            float val = m[i];
            NoteStatus& s = noteStates_[i];

            // Harmonic suppression: octave (12 semitones) and ~fifth harmonic (19)
            if (i >= 12 && m[i - 12] > val * 0.4f) val *= 0.1f;
            if (i >= 19 && m[i - 19] > val * 0.6f) val *= 0.2f;

            // Hysteresis: separate on/off thresholds
            if (!s.isActive) {
                if (val > dynamicOn) {
                    s.isActive       = true;
                    s.triggerPending = true;
                    float norm = std::clamp(val / (onThresh * 20.0f), 0.0f, 1.0f);
                    s.velocity = static_cast<int>(127.0f * std::pow(norm, 1.0f / sens));
                    s.velocity = std::max(10, s.velocity);
                }
            } else {
                if (val < offThresh) {
                    s.isActive       = false;
                    s.triggerPending = false;
                }
            }
            s.currentMag = val;
        }
    }

    // Reset all Goertzel state (call on onset to flush stale energy)
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
        }
    }

    int startMidi() const { return startMidi_; }
    int numNotes()  const { return numNotes_; }
    std::vector<NoteStatus>& getNoteStates() { return noteStates_; }
    const std::vector<NoteStatus>& getNoteStates() const { return noteStates_; }

private:
    float sampleRate_ = 48000.0f;
    int   startMidi_  = 40;
    int   numNotes_   = 0;
    int   numGroups_  = 0;
    std::vector<BinGroup>    groups_;
    std::vector<NoteStatus>  noteStates_;
    std::vector<float>       magBuf_;  // flat magnitude buffer for scalar access
};
