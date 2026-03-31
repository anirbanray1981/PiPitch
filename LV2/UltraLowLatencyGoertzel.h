#pragma once
/**
 * UltraLowLatencyGoertzel — sample-by-sample polyphonic pitch detection.
 *
 * Processes audio at native sample rate (48 kHz) using NEON SIMD.
 * 4 bins per SIMD lane × 13 groups = 52 slots (49 notes E2–C6 + 3 padding).
 *
 * Features:
 * - Onset blanking: freeze note detection for ~5ms after pick attack
 * - Multi-block magnitude accumulation: Hann window over 256-sample spans
 * - Frequency-scaled thresholds: low bins need higher energy to trigger
 * - Onset-aware dynamic threshold: 50ms elevated threshold after onset
 * - Exponential decay prevents unbounded energy accumulation
 * - Separate activeCount/inactiveCount debounce (clean hysteresis)
 * - Winner-takes-all per octave (strongest bin in 12-note range wins)
 * - Multi-interval harmonic suppression with absolute magnitude floor
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

    struct NoteState {
        int   activeCount   = 0;
        int   inactiveCount = 0;
        int   holdRemain    = 0;    // eval cycles remaining in hold (delays OFF)
        bool  isMidiOn      = false;
        bool  triggerPending = false;
        int   velocity       = 0;
        float currentMag     = 0.0f;

        // Aliases for external code
        bool isActive() const { return isMidiOn; }
    };

    static constexpr int   CONFIDENCE_BLOCKS = 3;       // blocks above threshold to trigger ON
    static constexpr int   HOLD_EVAL_CYCLES  = 6;       // ~30ms hold after note-ON (6 × 5.3ms)
    static constexpr int   MAX_BLOCK_SIZE    = 256;
    static constexpr float DECAY_FACTOR      = 0.99985f; // ~50ms half-life at 48kHz
    static constexpr float ON_THRESHOLD      = 0.08f;    // base magnitude for note-ON
    static constexpr float OFF_THRESHOLD     = 0.02f;    // magnitude for note-OFF
    static constexpr float HANN_ENERGY_COMP  = 2.0f;     // Hann ~50% energy loss compensation

    // Onset blanking: freeze detection for this many samples after onset
    static constexpr float ONSET_BLANK_MS    = 5.0f;
    // Onset-aware elevated threshold: scale ON_THRESHOLD by this during onset window
    static constexpr float ONSET_THRESH_MULT = 20.0f;
    static constexpr float ONSET_WINDOW_MS   = 200.0f;   // how long elevated threshold lasts
    // Multi-block accumulation: evaluate notes every N samples (not every block)
    static constexpr int   EVAL_INTERVAL     = 256;      // ~5.3ms at 48kHz
    // Harmonic suppression minimum magnitude floor — below this,
    // relative comparisons are unreliable (noise floor artifacts).
    static constexpr float HARMONIC_MAG_FLOOR = 0.1f;

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

        // Pre-compute frequency-scaled ON thresholds.
        // Low notes (E2=82Hz) need ~4× higher threshold than mid notes (E4=330Hz)
        // because their IIR resonators ring longer from impulse noise.
        noteOnThresh_.resize(numNotes_);
        for (int i = 0; i < numNotes_; ++i) {
            int midi = startMidi + i;
            float freq = 440.0f * std::pow(2.0f, (midi - 69) / 12.0f);
            // Scale factor: 1.0 at 330Hz (E4), rises quadratically for lower notes.
            // E2 (82Hz) gets ~16× higher threshold; E3 (165Hz) gets ~4×.
            float ratio = std::max(1.0f, 330.0f / freq);
            float scale = ratio * ratio;  // quadratic: low bins need much higher threshold
            noteOnThresh_[i] = ON_THRESHOLD * scale;
        }

        // Pre-compute Hann window for EVAL_INTERVAL
        evalHann_.resize(EVAL_INTERVAL);
        for (int i = 0; i < EVAL_INTERVAL; ++i)
            evalHann_[i] = 0.5f * (1.0f - std::cos(2.0f * M_PI * i / (EVAL_INTERVAL - 1)));

        // Accumulation buffer for multi-block evaluation
        accumBuf_.resize(EVAL_INTERVAL, 0.0f);
        accumPos_ = 0;

        onsetBlankSamples_  = static_cast<int>(fs * ONSET_BLANK_MS / 1000.0f);
        onsetWindowSamples_ = static_cast<int>(fs * ONSET_WINDOW_MS / 1000.0f);
        blankRemain_ = 0;
        onsetRemain_ = 0;

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

    /**
     * Process a block of audio.
     * @param input      Raw audio samples at native sample rate
     * @param nSamples   Block size (typically 64)
     * @param onsetFired True if a pick onset was detected THIS block
     */
    void processBlock(const float* input, int nSamples, bool onsetFired = false)
    {
        if (nSamples <= 0 || nSamples > MAX_BLOCK_SIZE) return;

        // Handle onset: start blanking + elevated threshold window
        if (onsetFired) {
            blankRemain_ = onsetBlankSamples_;
            onsetRemain_ = onsetWindowSamples_;
        }

        // Feed samples through Goertzel IIR (always — maintains state continuity)
        // but accumulate into eval buffer for windowed magnitude computation.
        for (int n = 0; n < nSamples; ++n) {
            const float sample = input[n];

            // Decrement blanking/onset counters
            if (blankRemain_ > 0) blankRemain_--;
            if (onsetRemain_ > 0) onsetRemain_--;

            // Always feed the IIR (even during blank — keeps filter state smooth)
#ifdef __aarch64__
            const float32x4_t inp = vdupq_n_f32(sample);
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
                    float s0 = sample + g.coeff[j] * g.s1[j] - g.s2[j];
                    s0 *= g.decay[j];
                    g.s2[j] = g.s1[j] * g.decay[j];
                    g.s1[j] = s0;
                }
            }
#endif

            // Accumulate into evaluation buffer
            accumBuf_[accumPos_++] = sample;

            // When eval buffer is full, compute magnitudes and run note detection
            if (accumPos_ >= EVAL_INTERVAL) {
                computeMagnitudes();
                if (blankRemain_ <= 0) {
                    updateNotes();
                }
                accumPos_ = 0;
            }
        }
    }

    // Overload for backward compatibility with transientRatio callers.
    // transientRatio > 1 means onset is active.
    void processBlock(const float* input, int nSamples, float transientRatio)
    {
        processBlock(input, nSamples, transientRatio > 1.5f);
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
            s.activeCount = s.inactiveCount = s.holdRemain = 0;
            s.isMidiOn = s.triggerPending = false;
            s.velocity = 0;
            s.currentMag = 0.0f;
        }
        accumPos_ = 0;
        blankRemain_ = 0;
        onsetRemain_ = 0;
        std::fill(accumBuf_.begin(), accumBuf_.end(), 0.0f);
    }

    int startMidi() const { return startMidi_; }
    int numNotes()  const { return numNotes_; }
    std::vector<NoteState>& getNoteStates() { return noteStates_; }
    const std::vector<NoteState>& getNoteStates() const { return noteStates_; }

private:
    /// Compute magnitudes from current IIR state.
    void computeMagnitudes()
    {
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
    }

    void updateNotes()
    {
        float* m = magBuf_.data();

        // Harmonic suppression — if a lower bin (potential fundamental) has
        // significant energy, suppress this bin as a likely harmonic.
        // Uses unmodified magnitudes for comparison (copy first).
        float raw[52];
        std::memcpy(raw, m, numNotes_ * sizeof(float));

        for (int i = 0; i < numNotes_; ++i) {
            float val = m[i];
            if (val < HARMONIC_MAG_FLOOR) continue;

            // Check against ORIGINAL magnitudes (raw[]) so earlier suppression
            // doesn't affect later comparisons.
            if (i >= 12 && raw[i - 12] > val * 0.15f) val *= 0.01f;  // octave
            if (i >= 24 && raw[i - 24] > val * 0.15f) val *= 0.01f;  // 2 octaves
            if (i >= 7  && raw[i - 7]  > val * 0.15f) val *= 0.02f;  // fifth
            if (i >= 19 && raw[i - 19] > val * 0.15f) val *= 0.02f;  // octave+fifth
            if (i >= 4  && raw[i - 4]  > val * 0.3f)  val *= 0.05f;  // major third
            if (i >= 5  && raw[i - 5]  > val * 0.3f)  val *= 0.05f;  // fourth
            if (i >= 16 && raw[i - 16] > val * 0.2f)  val *= 0.02f;  // octave+fourth
            if (i >= 28 && raw[i - 28] > val * 0.15f) val *= 0.01f;  // 2oct+third
            if (i >= 31 && raw[i - 31] > val * 0.15f) val *= 0.01f;  // 2oct+fifth (H6)
            m[i] = val;
        }

        // Winner-takes-all per octave
        for (int i = 0; i < numNotes_; ++i) {
            if (m[i] <= 0.0f) continue;
            const int lo = std::max(0, i - 6);
            const int hi = std::min(numNotes_ - 1, i + 5);
            float maxInOctave = 0.0f;
            for (int k = lo; k <= hi; ++k)
                if (k != i && m[k] > maxInOctave) maxInOctave = m[k];
            if (maxInOctave > 0.0f && m[i] < maxInOctave)
                m[i] = 0.0f;
        }

        // Onset-aware threshold: during the onset window (50ms after pick),
        // raise the ON threshold dramatically to reject broadband transient energy.
        // Ramps linearly from ONSET_THRESH_MULT down to 1.0 over the window.
        float onsetMult = 1.0f;
        if (onsetRemain_ > 0) {
            float progress = static_cast<float>(onsetRemain_) / static_cast<float>(onsetWindowSamples_);
            onsetMult = 1.0f + (ONSET_THRESH_MULT - 1.0f) * progress;
        }

        // Note tracking with frequency-scaled thresholds and hold timer
        for (int i = 0; i < numNotes_; ++i) {
            const float val = m[i];
            NoteState& s = noteStates_[i];
            s.currentMag = val;

            const float dynamicOn = noteOnThresh_[i] * onsetMult;

            if (val > dynamicOn) {
                s.activeCount++;
                s.inactiveCount = 0;
                if (s.isMidiOn)
                    s.holdRemain = HOLD_EVAL_CYCLES;  // refresh hold while active

                if (s.activeCount >= CONFIDENCE_BLOCKS && !s.isMidiOn) {
                    s.isMidiOn       = true;
                    s.triggerPending = true;
                    s.holdRemain     = HOLD_EVAL_CYCLES;
                    float norm = std::clamp(val / (noteOnThresh_[i] * 20.0f), 0.0f, 1.0f);
                    s.velocity = static_cast<int>(127.0f * std::pow(norm, 1.0f / 1.2f));
                    s.velocity = std::max(10, s.velocity);
                }
            } else if (val < OFF_THRESHOLD) {
                s.inactiveCount++;
                s.activeCount = 0;

                if (s.isMidiOn) {
                    if (s.holdRemain > 0) {
                        s.holdRemain--;  // hold — don't turn off yet
                    } else if (s.inactiveCount >= CONFIDENCE_BLOCKS) {
                        s.isMidiOn       = false;
                        s.triggerPending = false;
                    }
                }
            }
            // Between OFF_THRESHOLD and dynamicOn: hysteresis hold
        }
    }

    float sampleRate_ = 48000.0f;
    int   startMidi_  = 40;
    int   numNotes_   = 0;
    int   numGroups_  = 0;
    int   accumPos_   = 0;
    int   blankRemain_       = 0;   // samples remaining in onset blank
    int   onsetRemain_       = 0;   // samples remaining in elevated-threshold window
    int   onsetBlankSamples_ = 0;   // precomputed from ONSET_BLANK_MS
    int   onsetWindowSamples_= 0;   // precomputed from ONSET_WINDOW_MS
    std::vector<BinGroup>   groups_;
    std::vector<NoteState>  noteStates_;
    std::vector<float>      magBuf_;
    std::vector<float>      noteOnThresh_;  // per-bin ON threshold (frequency-scaled)
    std::vector<float>      evalHann_;      // Hann window for eval interval
    std::vector<float>      accumBuf_;      // accumulation buffer
};
