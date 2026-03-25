/**
 * NeuralNote Guitar2MIDI — Latency Benchmark
 *
 * Measures the wall-clock cost of BasicPitch::transcribeToMIDI() and
 * reports end-to-end note latency for common LV2 host block sizes.
 *
 * Usage:
 *   latency_bench <bundle_path> [iterations]
 *
 * Example:
 *   latency_bench build_lv2/neuralnote_guitar2midi.lv2 10
 *
 * The test signal is a 440 Hz (A4) sine wave at 22050 Hz — the engine's
 * native sample rate — so no resampling is involved in the benchmark.
 *
 * Latency model
 * ─────────────
 * The LV2 plugin accumulates audio into a ring buffer until it holds
 * AUDIO_WINDOW_LENGTH seconds (2 s, 44100 samples at 22050 Hz).  Only
 * then does it call transcribeToMIDI().  The resulting MIDI events are
 * queued and delivered at the START of the next run() call.
 *
 *   end-to-end latency ≈ buffering_window + inference_time + one_host_block
 *
 * Because the host may run at any sample rate and block size, the
 * buffering window as seen by the wall clock varies:
 *
 *   buffering_wall_ms = (44100 / host_sr) * 1000
 *                     ≈ 2000 ms  @  22050 Hz  (plugin native rate)
 *                     ≈ 918 ms   @  48000 Hz  (most common Pi / Linux rate)
 *                     ≈ 1000 ms  @  44100 Hz
 *
 * The streaming simulation below exercises the full resample + accumulate
 * + transcribe loop at several realistic host configurations.
 */

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <vector>

// BinaryData.h must come before any Lib/Model header
#include "BinaryData.h"
#include "BasicPitch.h"
#include "BasicPitchConstants.h"

// ── helpers ──────────────────────────────────────────────────────────────────

static constexpr double TWO_PI = 6.28318530717958647692528;

static std::vector<float> makeSine(double freqHz, double sr, double durationSec)
{
    const int n = static_cast<int>(sr * durationSec);
    std::vector<float> buf(n);
    for (int i = 0; i < n; ++i)
        buf[i] = 0.5f * static_cast<float>(std::sin(TWO_PI * freqHz / sr * i));
    return buf;
}

// Linear resampler matching neuralnote_impl.cpp resampleLinear()
static void resampleLinear(const float* in, int inLen, double srcRate,
                            std::vector<float>& out)
{
    if (srcRate == 22050.0) {
        out.insert(out.end(), in, in + inLen);
        return;
    }
    const double ratio  = 22050.0 / srcRate;
    const int    outLen = static_cast<int>(inLen * ratio);
    for (int i = 0; i < outLen; ++i) {
        const double srcPos = i / ratio;
        const int    s0     = static_cast<int>(srcPos);
        const double frac   = srcPos - s0;
        const int    s1     = std::min(s0 + 1, inLen - 1);
        out.push_back(static_cast<float>((1.0 - frac) * in[s0] + frac * in[s1]));
    }
}

using Clock = std::chrono::high_resolution_clock;
using Ms    = std::chrono::duration<double, std::milli>;

// ── streaming simulation ──────────────────────────────────────────────────────
// Models the exact run() loop in neuralnote_impl.cpp:
//   resample → accumulate → when accumulated ≥ BLOCK_SAMPLES: transcribe
//
// Returns the wall-clock latency (ms) from the first sample until the first
// MIDI note event is available, for a given host sample rate and block size.

struct SimResult {
    double totalLatencyMs;   // wall-clock time from first sample to first note
    double bufferLatencyMs;  // portion spent accumulating audio
    double inferenceMs;      // transcribeToMIDI() wall time
    int    runsUntilMidi;    // number of host run() calls until notes appear
    int    notesFound;
};

// Old sequential model: accumulate 2 s → transcribe → clear → repeat.
static SimResult simulateStreaming(BasicPitch& bp, double hostRate, int blockSize,
                                    double signalDurationSec = 4.0)
{
    auto hostSignal = makeSine(440.0, hostRate, signalDurationSec);

    bp.reset();
    bp.setParameters(0.7f, 0.5f, 50.0f);

    static constexpr int BLOCK_SAMPLES = AUDIO_SAMPLE_RATE * AUDIO_WINDOW_LENGTH;

    std::vector<float> accumulator;
    accumulator.reserve(BLOCK_SAMPLES + 4096);

    int    runsUntilMidi   = 0;
    double bufferLatencyMs = 0.0;
    double inferenceMs     = 0.0;
    bool   done            = false;

    const int totalSamples = static_cast<int>(hostSignal.size());
    const double msPerBlock = static_cast<double>(blockSize) / hostRate * 1000.0;

    for (int offset = 0; offset + blockSize <= totalSamples && !done;
         offset += blockSize)
    {
        ++runsUntilMidi;
        bufferLatencyMs += msPerBlock;
        resampleLinear(hostSignal.data() + offset, blockSize, hostRate, accumulator);

        if (static_cast<int>(accumulator.size()) >= BLOCK_SAMPLES) {
            auto t0 = Clock::now();
            bp.transcribeToMIDI(accumulator.data(), static_cast<int>(accumulator.size()));
            inferenceMs = Ms(Clock::now() - t0).count();
            accumulator.clear();
            if (!bp.getNoteEvents().empty()) done = true;
        }
    }

    const double totalLatencyMs = bufferLatencyMs + inferenceMs + msPerBlock;
    return { totalLatencyMs, bufferLatencyMs, inferenceMs,
             runsUntilMidi, static_cast<int>(bp.getNoteEvents().size()) };
}

// New streaming model: ring buffer + background worker.
//
// The worker clears workerHasWork before starting inference, so run() can
// queue a fresh snapshot immediately.  This means:
//
//   - A note played at an arbitrary point after the last snapshot was taken
//     will appear in the NEXT snapshot (queued within one host block ≈ 10 ms).
//   - That snapshot waits until the current inference finishes (up to
//     inference_ms).
//   - Then inference runs again (inference_ms).
//   - Total worst-case: 2 × inference_ms + one_block_ms
//   - Total best-case:  inference_ms + one_block_ms
//   - Average:         ~1.5 × inference_ms
//
// This function measures the three cases empirically on a real 2 s tone.
struct StreamingResult {
    double bestMs;
    double worstMs;
    double avgMs;
    double inferenceMs;
};

static StreamingResult simulateStreamingNew(double hostRate, int blockSize)
{
    static constexpr int RING = AUDIO_SAMPLE_RATE * AUDIO_WINDOW_LENGTH; // 44100
    const double msPerBlock   = static_cast<double>(blockSize) / hostRate * 1000.0;

    // Pre-fill a 2 s ring buffer with 440 Hz tone (represents audio already playing)
    auto native = makeSine(440.0, AUDIO_SAMPLE_RATE, AUDIO_WINDOW_LENGTH);

    // Measure one inference on a full ring
    BasicPitch bp;
    bp.setParameters(0.7f, 0.5f, 50.0f);

    auto t0 = Clock::now();
    bp.transcribeToMIDI(native.data(), RING);
    const double inf = Ms(Clock::now() - t0).count();

    // Worst case: note played at the moment inference starts.
    //   → Current snapshot misses note; next snapshot (queued in 1 block) is
    //     picked up after inference finishes; one more inference needed.
    const double worst = inf + msPerBlock + inf + msPerBlock;

    // Best case: note played just before snapshot is taken.
    //   → Note captured; inference runs; MIDI delivered in next block.
    const double best  = inf + msPerBlock;

    return { best, worst, (best + worst) / 2.0, inf };
}

// ── main ─────────────────────────────────────────────────────────────────────

int main(int argc, char** argv)
{
    const char* bundlePath = (argc >= 2) ? argv[1] : ".";
    const int   iterations = (argc >= 3) ? std::atoi(argv[2]) : 5;

    // ── load models ──────────────────────────────────────────────────────────
    printf("Loading models from: %s\n", bundlePath);
    try {
        BinaryData::init(bundlePath);
    } catch (const std::exception& e) {
        fprintf(stderr, "ERROR: %s\n", e.what());
        return 1;
    }
    printf("Models loaded.\n\n");

    // ── raw inference benchmark ───────────────────────────────────────────────
    // 2 s of 440 Hz at 22050 Hz (engine native rate — no resampling overhead)
    const int BLOCK = AUDIO_SAMPLE_RATE * AUDIO_WINDOW_LENGTH; // 44100 samples
    auto nativeSignal = makeSine(440.0, AUDIO_SAMPLE_RATE, AUDIO_WINDOW_LENGTH);

    // warm-up
    {
        BasicPitch bp;
        bp.setParameters(0.7f, 0.5f, 50.0f);
        bp.transcribeToMIDI(nativeSignal.data(), BLOCK);
        printf("Warm-up complete. Notes detected: %zu\n\n",
               bp.getNoteEvents().size());
    }

    printf("=== Raw inference benchmark (%d iterations, 22050 Hz, 2 s block) ===\n",
           iterations);

    std::vector<double> times;
    times.reserve(iterations);
    int totalNotes = 0;

    for (int i = 0; i < iterations; ++i) {
        BasicPitch bp;
        bp.setParameters(0.7f, 0.5f, 50.0f);

        auto t0 = Clock::now();
        bp.transcribeToMIDI(nativeSignal.data(), BLOCK);
        double ms = Ms(Clock::now() - t0).count();

        times.push_back(ms);
        int n = static_cast<int>(bp.getNoteEvents().size());
        totalNotes += n;
        printf("  [%2d]  %.1f ms   notes: %d\n", i + 1, ms, n);
    }

    const double sum  = std::accumulate(times.begin(), times.end(), 0.0);
    const double avg  = sum / iterations;
    const double minT = *std::min_element(times.begin(), times.end());
    const double maxT = *std::max_element(times.begin(), times.end());

    printf("\n  Inference time — min: %.1f ms  avg: %.1f ms  max: %.1f ms\n",
           minT, avg, maxT);
    printf("  Avg notes per block: %.1f\n", static_cast<double>(totalNotes) / iterations);

    // ── streaming latency simulation ─────────────────────────────────────────
    printf("\n=== Streaming latency simulation ===\n");
    printf("  (Signal: 440 Hz sine, 4 s long.  Latency = buffering + inference + 1 block.)\n\n");

    struct Config { double rate; int blockSize; const char* label; };
    static const Config configs[] = {
        { 22050,  512, "22050 Hz / 512 samples  (plugin native)" },
        { 44100,  512, "44100 Hz / 512 samples" },
        { 48000,  512, "48000 Hz / 512 samples  (Pisound default)" },
        { 48000, 1024, "48000 Hz / 1024 samples" },
        { 48000,  128, "48000 Hz / 128 samples  (low-latency)" },
        { 44100, 1024, "44100 Hz / 1024 samples" },
    };
    static const int nConfigs = static_cast<int>(sizeof(configs) / sizeof(configs[0]));

    printf("  %-45s  %8s  %8s  %8s  %7s  %6s\n",
           "Configuration", "Buffer", "Infer", "Total", "Runs", "Notes");
    printf("  %-45s  %8s  %8s  %8s  %7s  %6s\n",
           std::string(45, '-').c_str(),
           "  (ms)  ", "  (ms)  ", "  (ms)  ", "     ", "     ");

    for (int c = 0; c < nConfigs; ++c) {
        BasicPitch bp;
        SimResult r = simulateStreaming(bp, configs[c].rate, configs[c].blockSize);
        printf("  %-45s  %8.0f  %8.1f  %8.0f  %7d  %6d\n",
               configs[c].label,
               r.bufferLatencyMs,
               r.inferenceMs,
               r.totalLatencyMs,
               r.runsUntilMidi,
               r.notesFound);
    }

    printf("\n  Notes:\n");
    printf("  * 'Buffer' = wall time spent accumulating 2 s of 22050-Hz audio.\n");
    printf("  * 'Infer'  = transcribeToMIDI() wall time (this benchmark run).\n");
    printf("  * 'Total'  = Buffer + Infer + 1 host block (MIDI delivery delay).\n");

    // ── New streaming model ───────────────────────────────────────────────────
    printf("\n=== New streaming model (ring buffer + background worker) ===\n");
    printf("  Ring buffer always holds the latest 2 s.  Worker runs inference\n");
    printf("  continuously; run() queues a fresh snapshot each cycle (non-blocking).\n\n");

    printf("  %-45s  %8s  %8s  %8s  %8s\n",
           "Configuration", "Infer", "Best", "Worst", "Avg");
    printf("  %-45s  %8s  %8s  %8s  %8s\n",
           std::string(45, '-').c_str(),
           "  (ms)  ", "  (ms)  ", "  (ms)  ", "  (ms)  ");

    for (int c = 0; c < nConfigs; ++c) {
        StreamingResult r = simulateStreamingNew(configs[c].rate, configs[c].blockSize);
        printf("  %-45s  %8.1f  %8.0f  %8.0f  %8.0f\n",
               configs[c].label,
               r.inferenceMs,
               r.bestMs, r.worstMs, r.avgMs);
    }

    printf("\n  Notes:\n");
    printf("  * 'Infer'  = transcribeToMIDI() wall time measured on this run.\n");
    printf("  * 'Best'   = note played just before snapshot → 1 inference + 1 block.\n");
    printf("  * 'Worst'  = note played just as inference starts → 2 inferences + 2 blocks.\n");
    printf("  * 'Avg'    = (Best + Worst) / 2.\n");
    printf("  * Initial fill still takes 2 s; these figures apply once the ring is full.\n");

    return 0;
}
