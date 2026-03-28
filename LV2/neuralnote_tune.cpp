/**
 * neuralnote_tune — NeuralNote parameter sweep / latency measurement tool
 *
 * Threading model
 * ───────────────
 * One worker thread per note range.  All inter-thread communication is lockless:
 *
 *   jackProcess → worker :  SnapshotChannel  (SPSC, atomic ready + POSIX semaphore)
 *   worker → jackProcess :  MidiOutQueue     (SPSC ring buffer, atomic head/tail)
 *
 * jackProcess writes audio into each range's ring buffer and dispatches a snapshot
 * to the worker as soon as MIN_FRESH_SAMPLES (25 ms) of new audio has arrived and
 * the previous snapshot has been consumed.  Workers run inference independently and
 * push note events to their MidiOutQueues.  processSynth polls all queues each
 * JACK callback — no mutex anywhere on the hot path.
 *
 * Usage:
 *   neuralnote_tune [--bundle PATH] [--config PATH]
 *                   [--threshold 0.6] [--frame-threshold 0.5] [--mode poly|mono]
 *                   [--hold-cycles 2] [--gate 0.003] [--amp-floor 0.65] [--window 150]
 *                   [--waveform sine|saw|square]
 *                   [--attack MS] [--release MS] [--volume 0.3]
 *
 * Config file: neuralnote_tune.conf next to the binary (or --config PATH).
 */

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <csignal>
#include <map>
#include <memory>
#include <semaphore.h>
#include <set>
#include <string>
#include <thread>
#include <vector>

#include <jack/jack.h>

// MPM is always enabled in neuralnote_tune (Pi 5 target only).
// Must be defined before NeuralNoteShared.h so that RangeStateBase and all
// shared pipeline functions compile in the McLeod call-sites.
#define NEURALNOTE_ENABLE_MPM 1

#include "BasicPitchConstants.h"
#include "NeuralNoteShared.h"  // pulls in BinaryData.h, BasicPitch.h, NoteRangeConfig.h,
                               // OneBitPitchDetector.h, McLeodPitchDetector.h
#include "SwiftF0Detector.h"

// ── tune-only constants ────────────────────────────────────────────────────────

static constexpr int MAX_VOICES = 16;

// ── Note helpers ───────────────────────────────────────────────────────────────

static const char* NOTE_NAMES[12] = {"C","C#","D","D#","E","F","F#","G","G#","A","A#","B"};

static std::string midiToName(int midi)
{
    char buf[16];
    std::snprintf(buf, sizeof(buf), "%s%d", NOTE_NAMES[midi % 12], (midi / 12) - 1);
    return buf;
}

static float midiToFreq(int midi)
{
    return 440.0f * std::pow(2.0f, (midi - 69) / 12.0f);
}

// ── Linear resampler ───────────────────────────────────────────────────────────

static void resampleLinear(const float* in, int inLen, double srcRate,
                            std::vector<float>& out)
{
    if (srcRate == PLUGIN_SR) { out.insert(out.end(), in, in + inLen); return; }
    const double ratio  = PLUGIN_SR / srcRate;
    const int    outLen = static_cast<int>(inLen * ratio);
    for (int i = 0; i < outLen; ++i) {
        const double pos = i / ratio;
        const int    s0  = static_cast<int>(pos);
        const double f   = pos - s0;
        const int    s1  = std::min(s0 + 1, inLen - 1);
        out.push_back(static_cast<float>((1.0 - f) * in[s0] + f * in[s1]));
    }
}

// ── Synth ──────────────────────────────────────────────────────────────────────

enum class Waveform { SINE, SAW, SQUARE };

static float synthSample(Waveform w, double phase)
{
    switch (w) {
        case Waveform::SINE:   return std::sin(2.0 * M_PI * phase);
        case Waveform::SAW:    return 1.0f - 2.0f * static_cast<float>(phase);
        case Waveform::SQUARE: return phase < 0.5 ? 1.0f : -1.0f;
        default:               return 0.0f;
    }
}

struct SynthVoice {
    int    pitch = -1; float freq = 0.0f; double phase = 0.0;
    float  velocity = 0.0f; int state = 0; float envLevel = 0.0f;
};

// ── Per-range runtime state ────────────────────────────────────────────────────
// All common fields (including MidiOutQueue midiOut) live in RangeStateBase.

struct RangeState : RangeStateBase {
    // tune-specific fields
    int    provMidiPitch    = -1;   // set by OBP, consumed by processSynth this callback
    double provOnTimeMs     = 0.0;  // ms since startTime when provisional fired
    int    lastSwiftPrint   = -2;   // last SwiftF0 note printed (-1=silent, -2=none yet)
};

// ── Shared state ───────────────────────────────────────────────────────────────

struct Monitor {
    double   sampleRate     = 48000.0;
    float    gateFloor      = 0.003f;
    float    ampFloor       = 0.65f;
    float    threshold      = 0.6f;
    float    frameThreshold = 0.5f;
    PlayMode mode           = PlayMode::POLY;

    std::vector<std::unique_ptr<RangeState>> ranges;

    std::chrono::steady_clock::time_point startTime;

    Waveform   waveform  = Waveform::SINE;
    float      attackMs  = 10.0f;
    float      releaseMs = 400.0f;
    float      masterVol = 0.3f;
    SynthVoice voices[MAX_VOICES];

    jack_port_t* inPort  = nullptr;
    jack_port_t* outPort = nullptr;

    // Onset detector state (jackProcess thread only)
    float onsetSmoothedRms = 0.001f;
    float onsetBlankMs     = 25.0f;  // re-trigger suppression window (ms)
    int   onsetBlankRemain = 0;      // raw input samples remaining in blank period

    // HPF pick detector (jackProcess thread only)
    PickDetector pickDetector;

    // Onset-recency tracking (audio → worker, for decay-tail ghost suppression)
    std::atomic<uint64_t> totalSamples{0};
    std::atomic<uint64_t> lastOnsetSample{0};

    // Per-range OBP provisional note fired this onset (-1 if none); cleared on each onset.
    // Used for cross-range harmonic suppression.
    std::array<int, 8> onsetProvNotes = {-1,-1,-1,-1,-1,-1,-1,-1};

    std::unique_ptr<SwiftF0Detector> swiftF0;         // null if model not found
    float                            swiftF0Threshold = 0.5f;
    std::vector<float>               sf0Buf;          // worker-thread scratch: 16 kHz audio

    // Single worker thread shared across all ranges
    std::thread       workerThread;
    std::atomic<bool> workerQuit{false};
    sem_t             workerSem;
};

static Monitor*          g_mon = nullptr;
static std::atomic<bool> g_quit{false};
static void onSignal(int) { g_quit.store(true); }

// ── Worker thread ──────────────────────────────────────────────────────────────

// newBits: bitmap of CNN-detected notes; newVel[i]: velocity (0–127) for bit i.
// Logs CNN outcome vs provisional, then delegates the note ON/OFF/hold state
// machine to the shared applyNotesDiff helper.
static void applyRangeDiff(Monitor* m, RangeState& r,
                           uint64_t newBits, const int8_t* newVel,
                           double inferMs, int prov, bool mono)
{
    const double elapsed = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - m->startTime).count();

    if (prov != -1) {
        // Log CNN outcome vs provisional
        if (bmTest(newBits, prov)) {
            std::printf("[+%.3fs]  --   CNN confirmed OBP %-4s (%3d)"
                        "  [inf %4.0fms  range %s]\n",
                        elapsed, midiToName(prov).c_str(), prov,
                        inferMs, r.cfg.name.c_str());
        } else if (newBits) {
            std::printf("[+%.3fs]  --   CNN corrected OBP %-4s (%3d) →",
                        elapsed, midiToName(prov).c_str(), prov);
            for (uint64_t tmp = newBits; tmp; tmp &= tmp - 1)
                std::printf("  %-4s (%3d)", midiToName(NOTE_BASE + __builtin_ctzll(tmp)).c_str(),
                             NOTE_BASE + __builtin_ctzll(tmp));
            std::printf("  [inf %4.0fms  range %s]\n", inferMs, r.cfg.name.c_str());
        } else {
            std::printf("[+%.3fs]  --   CNN cancelled OBP %-4s (%3d)"
                        "  [inf %4.0fms  range %s]\n",
                        elapsed, midiToName(prov).c_str(), prov,
                        inferMs, r.cfg.name.c_str());
        }
        std::fflush(stdout);
    }

    // Snapshot active state for post-diff logging
    const uint64_t preActive = r.activeNotes;

    // Shared note ON/OFF/hold state machine (blacklist, force-expire, push events)
    applyNotesDiff(r, newBits, newVel, prov, mono);

    // Log note-ONs (notes that became active this cycle)
    for (uint64_t tmp = r.activeNotes & ~preActive; tmp; tmp &= tmp - 1) {
        const int bit = __builtin_ctzll(tmp);
        const int p   = NOTE_BASE + bit;
        std::printf("[+%.3fs]  ON   %-4s (%3d)  vel %3d"
                    "  [CNN  win %4.0fms  inf %4.0fms  range %s]\n",
                    elapsed, midiToName(p).c_str(), p, newVel[bit],
                    r.cfg.windowMs, inferMs, r.cfg.name.c_str());
        std::fflush(stdout);
    }
    // Log note-OFFs (notes that left active this cycle)
    for (uint64_t tmp = preActive & ~r.activeNotes; tmp; tmp &= tmp - 1) {
        const int p = NOTE_BASE + static_cast<int>(__builtin_ctzll(tmp));
        std::printf("[+%.3fs]  OFF  %-4s (%3d)\n", elapsed, midiToName(p).c_str(), p);
        std::fflush(stdout);
    }
}

static void runWorker(Monitor* m)
{
    while (true) {
        sem_wait(&m->workerSem);
        if (m->workerQuit.load(std::memory_order_acquire)) break;

        const bool swiftMono = (m->mode == PlayMode::SWIFT_MONO && m->swiftF0 != nullptr);
        const bool mono      = (m->mode == PlayMode::MONO || m->mode == PlayMode::SWIFT_MONO);

        for (int ri = 0; ri < static_cast<int>(m->ranges.size()); ++ri) {
            RangeState& r = *m->ranges[ri];
            if (!r.snapChan.ready.load(std::memory_order_acquire)) continue;

            // Always update BasicPitch params (no-op in swiftmono but keeps state clean).
            const float minDurMs = r.cfg.minNoteLength * FFT_HOP
                                   / static_cast<float>(PLUGIN_SR) * 1000.0f;
            r.basicPitch->setParameters(1.0f - m->frameThreshold, m->threshold, minDurMs);

            // Two-phase: insert provisional into activeNotes before diff.
            // Staleness check: if a new provisional has fired since this snapshot
            // was dispatched, don't cancel it — the CNN result doesn't correspond
            // to the current provisional.
            const int snapProv = r.snapChan.provNoteAtDispatch;
            r.snapChan.provNoteAtDispatch = -1;

            int provForDiff = -1;
            if (snapProv != -1) {
                const int currentProv = r.provNote.load(std::memory_order_acquire);
                if (currentProv == snapProv) {
                    r.provNote.store(-1, std::memory_order_release);
                    if (snapProv >= NOTE_BASE && snapProv < NOTE_BASE + NOTE_COUNT) {
                        r.activeNotes |= (1ULL << (snapProv - NOTE_BASE));
                        provForDiff = snapProv;
                    } else {
                        r.midiOut.push({false, snapProv, 0});
                    }
                }
                // If stale: leave provNote intact; provForDiff stays -1.
            }

            uint64_t newBits = 0;
            int8_t   newVel[NOTE_COUNT] = {};
            double   inferMs = 0.0;

            if (swiftMono) {
                // SwiftF0 path: resample snapshot 22050→16000 Hz, run inference.
                const float* src    = r.snapChan.data.data();
                const int    srcLen = r.snapChan.snapshotSize;
                const bool   wasOnset = r.snapChan.onsetDispatched;
                constexpr double SF0_SR = 16000.0;
                const double ratio  = SF0_SR / PLUGIN_SR;
                const int    dstLen = static_cast<int>(srcLen * ratio);
                m->sf0Buf.resize(dstLen);
                for (int i = 0; i < dstLen; ++i) {
                    const double pos = i / ratio;
                    const int    s0  = static_cast<int>(pos);
                    const double f   = pos - s0;
                    const int    s1  = std::min(s0 + 1, srcLen - 1);
                    m->sf0Buf[i]     = static_cast<float>((1.0 - f) * src[s0] + f * src[s1]);
                }
                r.snapChan.ready.store(false, std::memory_order_release);

                const auto t0 = std::chrono::steady_clock::now();
                const int midiNote = m->swiftF0->infer(
                    m->sf0Buf.data(), dstLen, m->swiftF0Threshold);
                inferMs = std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - t0).count();

                int effectiveNote = -1;  // -1 = silent
                if (midiNote >= r.cfg.midiLow && midiNote <= r.cfg.midiHigh &&
                    midiNote >= NOTE_BASE && midiNote < NOTE_BASE + NOTE_COUNT) {
                    effectiveNote = midiNote;
                    const int bit = midiNote - NOTE_BASE;
                    newBits    = 1ULL << bit;
                    newVel[bit] = 100;
                }

                // Onset-transition grace: the first snapshot(s) after an onset
                // contain stale ring audio.  Two cases:
                //   Mid-note: keep current note for 1 cycle (half-step suppression).
                //   From silence: suppress cycle 1 and remember the stale note;
                //     cycle 2 allows immediately if a DIFFERENT note is detected
                //     (it's the real new note), or suppresses if same stale note.
                if (wasOnset && r.swiftOnsetGrace <= 0) {
                    r.swiftOnsetGrace = 2;
                    r.swiftGraceStaleNote = -1;
                }
                if (r.swiftOnsetGrace > 0) {
                    if (r.activeNotes && newBits && newBits != r.activeNotes) {
                        // Mid-note transition: wrong note → keep current
                        newBits = r.activeNotes;
                        effectiveNote = NOTE_BASE + __builtin_ctzll(r.activeNotes);
                        for (uint64_t tmp = newBits; tmp; tmp &= tmp - 1)
                            newVel[__builtin_ctzll(tmp)] = 100;
                    } else if (!r.activeNotes && newBits) {
                        const int detected = NOTE_BASE + __builtin_ctzll(newBits);
                        if (r.swiftGraceStaleNote < 0) {
                            r.swiftGraceStaleNote = detected;
                            newBits = 0;
                            effectiveNote = -1;
                        } else if (detected == r.swiftGraceStaleNote) {
                            newBits = 0;
                            effectiveNote = -1;
                        }
                    }
                    --r.swiftOnsetGrace;
                }

                // Decay-tail ghost suppression: if SwiftF0 wants to create a
                // new note-ON but no onset fired recently, this is likely a
                // harmonic from a decaying string.  In mono/swiftmono every
                // legitimate new note starts with an onset.
                if (newBits & ~r.activeNotes) {
                    const uint64_t elapsed =
                        m->totalSamples.load(std::memory_order_relaxed)
                        - m->lastOnsetSample.load(std::memory_order_acquire);
                    constexpr double ONSET_GATE_S = 0.25;
                    const auto gate = static_cast<uint64_t>(m->sampleRate * ONSET_GATE_S);
                    if (elapsed > gate) {
                        newBits = 0;
                        effectiveNote = -1;
                    }
                }

                // Only print when the detected note changes for this range.
                if (effectiveNote != r.lastSwiftPrint) {
                    r.lastSwiftPrint = effectiveNote;
                    const double elapsed = std::chrono::duration<double>(
                        std::chrono::steady_clock::now() - m->startTime).count();
                    if (effectiveNote >= 0) {
                        std::printf("[+%.3fs]  --   SwiftF0 → %-4s (%3d)"
                                    "  [inf %4.0fms  range %s]\n",
                                    elapsed, midiToName(effectiveNote).c_str(),
                                    effectiveNote, inferMs, r.cfg.name.c_str());
                    } else {
                        std::printf("[+%.3fs]  --   SwiftF0 → silent"
                                    "  [inf %4.0fms  range %s]\n",
                                    elapsed, inferMs, r.cfg.name.c_str());
                    }
                    std::fflush(stdout);
                }
            } else {
                // BasicPitch (CNN) path.
                const auto t0 = std::chrono::steady_clock::now();
                r.basicPitch->transcribeToMIDI(r.snapChan.data.data(), r.snapChan.snapshotSize);
                r.snapChan.ready.store(false, std::memory_order_release);
                inferMs = std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - t0).count();
                buildNNBits(r, m->ampFloor, newBits, newVel);
            }

            const uint64_t prevActive = r.activeNotes;

            // Cancel grace: first inference cycle after a new provisional fires often
            // contains mostly the previous note's audio.  Suppress that cancel so
            // the next cycle (with a full window of the new note) decides instead.
            // Re-enabled for swiftmono: now that the immediate path requires
            // OBP==MPM agreement, provisionals are accurate and worth protecting.
            {
                if (provForDiff != -1 && provForDiff != r.provLastSeenByCNN) {
                    r.provLastSeenByCNN = provForDiff;
                    r.provCancelGrace   = 1;
                }
                if (provForDiff != -1 && r.provCancelGrace > 0) {
                    const int bit = provForDiff - NOTE_BASE;
                    if (bit >= 0 && bit < NOTE_COUNT && !(newBits & (1ULL << bit))) {
                        --r.provCancelGrace;
                        newBits    |= (1ULL << bit);  // suppress cancel; note already playing
                        newVel[bit] = 64;             // velocity unused (no new note-ON fired)
                    } else {
                        r.provCancelGrace = 0;  // confirmed naturally
                    }
                }
            }

            if (swiftMono) {
                // SwiftF0 path: call state machine directly (SwiftF0 log already printed above).
                applyNotesDiff(r, newBits, newVel, provForDiff, mono,
                               r.cfg.swiftHoldCycles);
                const double elapsed = std::chrono::duration<double>(
                    std::chrono::steady_clock::now() - m->startTime).count();
                for (uint64_t tmp = r.activeNotes & ~prevActive; tmp; tmp &= tmp - 1) {
                    const int bit = __builtin_ctzll(tmp);
                    const int p   = NOTE_BASE + bit;
                    std::printf("[+%.3fs]  ON   %-4s (%3d)  vel %3d"
                                "  [SwiftF0  win %4.0fms  inf %4.0fms  range %s]\n",
                                elapsed, midiToName(p).c_str(), p, newVel[bit],
                                r.cfg.windowMs, inferMs, r.cfg.name.c_str());
                }
                for (uint64_t tmp = prevActive & ~r.activeNotes; tmp; tmp &= tmp - 1) {
                    const int p = NOTE_BASE + static_cast<int>(__builtin_ctzll(tmp));
                    std::printf("[+%.3fs]  OFF  %-4s (%3d)\n",
                                elapsed, midiToName(p).c_str(), p);
                }
                std::fflush(stdout);
            } else {
                applyRangeDiff(m, r, newBits, newVel, inferMs, provForDiff, mono);
            }

            // Mono/SwiftMono cross-range: new note-ON(s) in this range → kill all other ranges
            if (mono && (r.activeNotes & ~prevActive)) {
                const double elapsed = std::chrono::duration<double>(
                    std::chrono::steady_clock::now() - m->startTime).count();
                for (int oi = 0; oi < static_cast<int>(m->ranges.size()); ++oi) {
                    if (oi == ri) continue;
                    RangeState& other = *m->ranges[oi];
                    for (uint64_t tmp = other.activeNotes; tmp; tmp &= tmp - 1) {
                        const int p = NOTE_BASE + static_cast<int>(__builtin_ctzll(tmp));
                        std::printf("[+%.3fs]  OFF  %-4s (%3d)  [mono kill  range %s]\n",
                                    elapsed, midiToName(p).c_str(), p, other.cfg.name.c_str());
                        other.midiOut.push({false, p, 0});
                    }
                    other.activeNotes = 0;
                    other.holdNotes   = 0;
                    std::memset(other.holdCounts, 0, NOTE_COUNT);
                    other.monoHeldNote.store(-1, std::memory_order_release);
                }
                std::fflush(stdout);
            }
        }
    }

    // Shutdown: release all active notes across all ranges
    const double elapsed = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - m->startTime).count();
    for (auto& rp : m->ranges) {
        RangeState& r = *rp;
        for (uint64_t tmp = r.activeNotes; tmp; tmp &= tmp - 1) {
            const int p = NOTE_BASE + static_cast<int>(__builtin_ctzll(tmp));
            std::printf("[+%.3fs]  OFF  %-4s (%3d)  [shutdown]\n",
                        elapsed, midiToName(p).c_str(), p);
            r.midiOut.push({false, p, 0});
        }
        r.activeNotes = 0;
        r.holdNotes   = 0;
    }
}

// ── Synth engine ──────────────────────────────────────────────────────────────

static void processSynth(Monitor* m, float* out, int nFrames)
{
    // Handle provisional OneBitPitch note-ONs with cross-range harmonic suppression.
    // Rule: when range[i] fires note P, suppress range[j] if it already fired P+12/24
    // (our note is its harmonic) or |P_j - P| ≤ 1 (adjacent-note interference).
    // Retroactively kill the synth voice for already-playing wrong-range provisionals.
    const double elapsed = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - m->startTime).count();
    const int nRanges = static_cast<int>(m->ranges.size());
    for (int ri = 0; ri < nRanges; ++ri) {
        RangeState& rp = *m->ranges[ri];
        const int pp = rp.provMidiPitch;
        if (pp == -1) continue;
        rp.provMidiPitch = -1;

        // Skip if the same note is already active — avoids redundant re-triggers
        // from RMS onset re-fires on the same note's energy.
        if (pp >= NOTE_BASE && pp < NOTE_BASE + NOTE_COUNT) {
            const uint64_t bits = rp.activeNotesBits.load(std::memory_order_acquire);
            if (bits & (1ULL << (pp - NOTE_BASE))) continue;
        }

        // Mono: suppress if ANY other range already fired a provisional this onset.
        // Poly: suppress harmonics (diff 12/24) and adjacent artefacts (diff ≤ 5).
        const bool mono = (m->mode == PlayMode::MONO || m->mode == PlayMode::SWIFT_MONO);
        bool suppressed = false;
        for (int j = 0; j < ri && !suppressed; ++j) {
            const int pj = m->onsetProvNotes[j];
            if (pj == -1) continue;
            if (mono) { suppressed = true; break; }
            const int diff = pp - pj;
            if (diff == 12 || diff == 24 || std::abs(diff) <= 5)
                suppressed = true;
        }
        if (suppressed) {
            rp.obdOnsetActive = false;  // don't retry for this onset
            continue;
        }

        // Record this provisional; retroactively silence any higher-range voice that
        // fired a harmonic or adjacent note earlier in this onset window.
        if (ri < (int)m->onsetProvNotes.size())
            m->onsetProvNotes[ri] = pp;
        for (int j = ri + 1; j < nRanges; ++j) {
            const int pj = (j < (int)m->onsetProvNotes.size()) ? m->onsetProvNotes[j] : -1;
            if (pj == -1) continue;
            const int diff = pj - pp;
            if (diff == 12 || diff == 24 || std::abs(diff) <= 5) {
                // Start release on the voice that was playing the wrong provisional
                for (auto& v : m->voices)
                    if (v.pitch == pj && v.state != 0 && v.state != 3)
                        v.state = 3;
                m->onsetProvNotes[j] = -1;
                m->ranges[j]->obdOnsetActive = false;
            }
        }

        rp.provOnTimeMs = elapsed * 1000.0;  // ms since startTime

        // Mono/SwiftMono: release all currently playing voices before firing new note
        if (m->mode == PlayMode::MONO || m->mode == PlayMode::SWIFT_MONO) {
            for (auto& v : m->voices)
                if (v.state != 0 && v.state != 3)
                    v.state = 3;  // enter release
        }

        // Re-hit: release existing voice for this pitch so the synth retriggering cleanly.
        for (auto& v : m->voices)
            if (v.pitch == pp && v.state != 0 && v.state != 3)
                v.state = 3;

        std::printf("[+%.3fs]  ON   %-4s (%3d)  vel 100"
                    "  [1-bit provisional  range %s]\n",
                    elapsed, midiToName(pp).c_str(), pp, rp.cfg.name.c_str());
        std::fflush(stdout);
        int   best      = 0;
        float bestLevel = m->voices[0].envLevel + (m->voices[0].state != 0 ? 1.0f : 0.0f);
        for (int i = 0; i < MAX_VOICES; ++i) {
            if (m->voices[i].state == 0) { best = i; break; }
            const float l = m->voices[i].envLevel;
            if (l < bestLevel) { bestLevel = l; best = i; }
        }
        m->voices[best] = { pp, midiToFreq(pp), 0.0, 100.0f / 127.0f, 1, 0.0f };
    }

    // Drain all ranges' MIDI output queues — lockless
    for (auto& rp : m->ranges) {
        PendingNote pn;
        while (rp->midiOut.pop(pn)) {
            if (pn.noteOn) {
                std::printf("[+%.3fs]  >>   %-4s (%3d)  vel %3d"
                            "  [synth ON   range %s]\n",
                            elapsed, midiToName(pn.pitch).c_str(),
                            pn.pitch, pn.velocity, rp->cfg.name.c_str());
                std::fflush(stdout);
                int   best      = 0;
                float bestLevel = m->voices[0].envLevel + (m->voices[0].state != 0 ? 1.0f : 0.0f);
                for (int i = 0; i < MAX_VOICES; ++i) {
                    if (m->voices[i].state == 0) { best = i; break; }
                    const float l = m->voices[i].envLevel;
                    if (l < bestLevel) { bestLevel = l; best = i; }
                }
                m->voices[best] = { pn.pitch, midiToFreq(pn.pitch), 0.0, pn.velocity / 127.0f, 1, 0.0f };
            } else {
                std::printf("[+%.3fs]  >>   %-4s (%3d)          "
                            "  [synth OFF  range %s]\n",
                            elapsed, midiToName(pn.pitch).c_str(),
                            pn.pitch, rp->cfg.name.c_str());
                std::fflush(stdout);
                for (int i = 0; i < MAX_VOICES; ++i)
                    if (m->voices[i].pitch == pn.pitch && m->voices[i].state != 0)
                        m->voices[i].state = 3;
            }
        }
    }

    const float attackRate  = 1.0f / std::max(1.0f, m->attackMs  * 0.001f * (float)m->sampleRate);
    const float releaseRate = 1.0f / std::max(1.0f, m->releaseMs * 0.001f * (float)m->sampleRate);
    const float scale       = m->masterVol / static_cast<float>(MAX_VOICES);
    std::memset(out, 0, nFrames * sizeof(float));

    for (int vi = 0; vi < MAX_VOICES; ++vi) {
        SynthVoice& v = m->voices[vi];
        if (v.state == 0) continue;
        const double phaseInc = v.freq / m->sampleRate;
        for (int i = 0; i < nFrames; ++i) {
            if      (v.state == 1) { v.envLevel += attackRate;  if (v.envLevel >= 1.0f) { v.envLevel = 1.0f; v.state = 2; } }
            else if (v.state == 3) { v.envLevel -= releaseRate; if (v.envLevel <= 0.0f) { v.envLevel = 0.0f; v.state = 0; } }
            if (v.state == 0) break;
            out[i] += synthSample(m->waveform, v.phase) * v.envLevel * v.velocity * scale;
            v.phase += phaseInc; if (v.phase >= 1.0) v.phase -= 1.0;
        }
    }
}

// ── JACK process callback ─────────────────────────────────────────────────────

struct JackCtx { Monitor* mon; };

static int jackProcess(jack_nframes_t nFrames, void* arg)
{
    Monitor* m = static_cast<JackCtx*>(arg)->mon;

    const float* in  = static_cast<const float*>(jack_port_get_buffer(m->inPort,  nFrames));
    float*       out = static_cast<float*>       (jack_port_get_buffer(m->outPort, nFrames));

    float sumSq = 0.0f;
    for (jack_nframes_t i = 0; i < nFrames; ++i) sumSq += in[i] * in[i];
    const float blockRms = std::sqrt(sumSq / static_cast<float>(nFrames));
    const bool  gated    = (blockRms < m->gateFloor);

    // Onset detection: HPF pick detector is primary, RMS is fallback.
    m->totalSamples.fetch_add(nFrames, std::memory_order_relaxed);
    bool onsetFired = false;

    // Primary: HPF pick detector with two-tier confirmation.
    //   Tier 1 (ratio ≥ 10): high confidence — immediate onset.
    //   Tier 2 (ratio 4–10): tentative — confirmed only if RMS also agrees.
    constexpr float PICK_HIGH_TIER = 10.0f;
    int  pickSample   = -1;
    float pickRatioVal = 0.0f;
    if (!gated)
        pickSample = m->pickDetector.process(in, static_cast<int>(nFrames), pickRatioVal);

    const bool rmsWouldFire = !gated && m->onsetBlankRemain <= 0
                              && blockRms > m->onsetSmoothedRms * ONSET_RATIO;

    if (pickSample >= 0 && (pickRatioVal >= PICK_HIGH_TIER || rmsWouldFire)) {
        onsetFired          = true;
        m->onsetBlankRemain = static_cast<int>(m->sampleRate * (m->onsetBlankMs / 1000.0f));
        m->onsetSmoothedRms = blockRms;
        m->lastOnsetSample.store(
            m->totalSamples.load(std::memory_order_relaxed)
                - static_cast<uint64_t>(nFrames) + static_cast<uint64_t>(pickSample),
            std::memory_order_release);
        m->onsetProvNotes.fill(-1);
        const double elapsed = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - m->startTime).count();
        const double offsetMs = pickSample / m->sampleRate * 1000.0;
        std::printf("[+%.3fs]  --   PICK onset  (sample %d/%u  +%.1fms  ratio %.1f%s)\n",
                    elapsed, pickSample, nFrames, offsetMs, pickRatioVal,
                    pickRatioVal < PICK_HIGH_TIER ? "  +RMS" : "");
        std::fflush(stdout);
    }

    // Fallback: RMS onset (hammer-ons, volume swells, etc.)
    if (!onsetFired) {
        if (m->onsetBlankRemain > 0) {
            m->onsetBlankRemain -= static_cast<int>(nFrames);
            if (m->onsetBlankRemain < 0) m->onsetBlankRemain = 0;
        } else if (rmsWouldFire) {
            onsetFired            = true;
            m->onsetBlankRemain   = static_cast<int>(m->sampleRate * (m->onsetBlankMs / 1000.0f));
            m->onsetSmoothedRms   = blockRms;
            m->lastOnsetSample.store(m->totalSamples.load(std::memory_order_relaxed),
                                     std::memory_order_release);
            m->onsetProvNotes.fill(-1);
            const double elapsed = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - m->startTime).count();
            std::printf("[+%.3fs]  --   RMS onset  (rms %.4f / bg %.4f)\n",
                        elapsed, blockRms, m->onsetSmoothedRms / ONSET_RATIO);
            std::fflush(stdout);
        }
    }
    if (!onsetFired && m->onsetBlankRemain == 0)
        m->onsetSmoothedRms = m->onsetSmoothedRms * (1.0f - ONSET_ALPHA) + blockRms * ONSET_ALPHA;

    // Resample once, shared across all ranges
    std::vector<float> resampled;
    resampled.reserve(static_cast<int>(nFrames * (PLUGIN_SR / m->sampleRate)) + 2);
    if (gated) resampled.assign(static_cast<int>(nFrames * (PLUGIN_SR / m->sampleRate)), 0.0f);
    else       resampleLinear(in, static_cast<int>(nFrames), m->sampleRate, resampled);

    for (auto& rp : m->ranges) {
        RangeState& r = *rp;

        pushRingSamples(r, resampled.data(), static_cast<int>(resampled.size()));

        // Two-phase OneBitPitch + MPM: arm on onset, expire after 100 ms.
        armOrExpireOBP(r, static_cast<float>(m->sampleRate),
                       static_cast<int>(nFrames), onsetFired);

        // Push to MPM while OBP window is active OR while awaiting MPM on a pending vote.
        if (!gated && (r.obdOnsetActive || r.obdPendingNote != -1))
            r.mpm.push(in, static_cast<int>(nFrames));

        if (r.obdOnsetActive && !gated) {
            if (r.provNote.load(std::memory_order_relaxed) == -1) {
                const int finalNote = runOBPHPS(r, in,
                                                static_cast<int>(nFrames),
                                                static_cast<float>(m->sampleRate),
                                                m->ranges);
                if (finalNote == -1 && !r.obdOnsetActive) {
                    // OBP expired — try MPM alone as fallback
                    const float  sr      = static_cast<float>(m->sampleRate);
                    const int    mpmNote = r.mpm.analyze(sr, r.cfg.midiLow, r.cfg.midiHigh);
                    const int    mpmFill = r.mpm.circFill;
                    const double elapsed = std::chrono::duration<double>(
                        std::chrono::steady_clock::now() - m->startTime).count();
                    if (mpmNote != -1) {
                        // Mono: suppress if another range already has a provisional
                        const bool mono = (m->mode == PlayMode::MONO || m->mode == PlayMode::SWIFT_MONO);
                        bool suppress = false;
                        if (mono) {
                            for (const auto& orp : m->ranges) {
                                if (orp.get() == &r) continue;
                                if (orp->activeNotesBits.load(std::memory_order_acquire) != 0
                                    || orp->provNote.load(std::memory_order_relaxed) != -1)
                                    { suppress = true; break; }
                            }
                        }
                        if (!suppress) {
                            std::printf("[+%.3fs]  --   MPM fallback (fill %d): %-4s(%d)"
                                        "  prov fired  [range %s]\n",
                                        elapsed, mpmFill,
                                        midiToName(mpmNote).c_str(), mpmNote,
                                        r.cfg.name.c_str());
                            std::fflush(stdout);
                            r.provMidiPitch = mpmNote;
                            r.provNote.store(mpmNote, std::memory_order_release);
                            r.monoHeldNote.store(mpmNote, std::memory_order_release);
                        } else {
                            std::printf("[+%.3fs]  --   OBP expired  MPM→%-4s(%d)"
                                        "  suppressed (mono)  [range %s]\n",
                                        elapsed, midiToName(mpmNote).c_str(), mpmNote,
                                        r.cfg.name.c_str());
                            std::fflush(stdout);
                        }
                    } else {
                        std::printf("[+%.3fs]  --   OBP window expired  no vote  [range %s]\n",
                                    elapsed, r.cfg.name.c_str());
                        std::fflush(stdout);
                    }
                }
                if (finalNote != -1) {
                    const float  sr      = static_cast<float>(m->sampleRate);
                    const int    mpmNote = r.mpm.analyze(sr, r.cfg.midiLow, r.cfg.midiHigh);
                    const int    mpmFill = r.mpm.circFill;
                    const double elapsed = std::chrono::duration<double>(
                        std::chrono::steady_clock::now() - m->startTime).count();
                    if (mpmNote == -1) {
                        // MPM not ready — save vote and retry next callbacks
                        r.obdPendingNote   = finalNote;
                        r.obdPendingRemain = static_cast<int>(m->sampleRate * 0.1f);
                        std::printf("[+%.3fs]  --   MPM not ready (fill %d/%d)"
                                    "  OBP→%-4s(%d)  pending  [range %s]\n",
                                    elapsed, mpmFill, McLeodPitchDetector::MPM_BUFSIZE,
                                    midiToName(finalNote).c_str(), finalNote,
                                    r.cfg.name.c_str());
                        std::fflush(stdout);
                    } else if (mpmNote == finalNote) {
                        // OBP + MPM agree — fire immediately
                        std::printf("[+%.3fs]  --   OBP+MPM agree (fill %d): %-4s(%d)"
                                    "  prov fired  [range %s]\n",
                                    elapsed, mpmFill,
                                    midiToName(finalNote).c_str(), finalNote,
                                    r.cfg.name.c_str());
                        std::fflush(stdout);
                        r.provMidiPitch = finalNote;
                        r.provNote.store(finalNote, std::memory_order_release);
                        r.monoHeldNote.store(finalNote, std::memory_order_release);
                    } else {
                        // OBP + MPM disagree — save pending; retry with more data
                        r.obdPendingNote   = finalNote;
                        r.obdPendingRemain = static_cast<int>(m->sampleRate * 0.1f);
                        std::printf("[+%.3fs]  --   MPM disagrees (fill %d):"
                                    "  OBP→%-4s(%d)  MPM→%-4s(%d)  pending  [range %s]\n",
                                    elapsed, mpmFill,
                                    midiToName(finalNote).c_str(), finalNote,
                                    midiToName(mpmNote).c_str(), mpmNote,
                                    r.cfg.name.c_str());
                        std::fflush(stdout);
                    }
                }
            }
        } else if (!gated && r.obdPendingNote != -1
                   && r.provNote.load(std::memory_order_relaxed) == -1) {
            // OBP voted previously but MPM wasn't ready — retry now.
            r.obdPendingRemain -= static_cast<int>(nFrames);
            if (r.obdPendingRemain <= 0) {
                const double elapsed = std::chrono::duration<double>(
                    std::chrono::steady_clock::now() - m->startTime).count();
                std::printf("[+%.3fs]  --   MPM pending timed out  OBP→%-4s(%d)"
                            "  [range %s]\n",
                            elapsed, midiToName(r.obdPendingNote).c_str(),
                            r.obdPendingNote, r.cfg.name.c_str());
                std::fflush(stdout);
                r.obdPendingNote = -1;
            } else {
                const float  sr      = static_cast<float>(m->sampleRate);
                const int    mpmNote = r.mpm.analyze(sr, r.cfg.midiLow, r.cfg.midiHigh);
                const int    mpmFill = r.mpm.circFill;
                if (mpmNote != -1) {
                    const int    pending = r.obdPendingNote;
                    const double elapsed = std::chrono::duration<double>(
                        std::chrono::steady_clock::now() - m->startTime).count();
                    r.obdPendingNote = -1;
                    // Mono: suppress if any other range has a provisional.
                    // Poly: suppress harmonics (±12/24).
                    const bool mono = (m->mode == PlayMode::MONO || m->mode == PlayMode::SWIFT_MONO);
                    bool mpmHarmonic = false;
                    for (const auto& orp : m->ranges) {
                        if (orp.get() == &r) continue;
                        if (mono && (orp->activeNotesBits.load(std::memory_order_acquire) != 0
                                     || orp->provNote.load(std::memory_order_relaxed) != -1))
                            { mpmHarmonic = true; break; }
                        const int op2 = orp->provNote.load(std::memory_order_relaxed);
                        if (op2 != -1) {
                            const int ad = std::abs(mpmNote - op2);
                            if (ad == 12 || ad == 24) { mpmHarmonic = true; break; }
                        }
                    }
                    if (mpmHarmonic) {
                        std::printf("[+%.3fs]  --   MPM→%-4s(%d) suppressed (mono/harmonic)"
                                    "  [range %s]\n",
                                    elapsed, midiToName(mpmNote).c_str(), mpmNote,
                                    r.cfg.name.c_str());
                        std::fflush(stdout);
                    } else {
                        if (mpmNote != pending) {
                            std::printf("[+%.3fs]  --   MPM corrects pending OBP→%-4s(%d)"
                                        "  MPM→%-4s(%d)  (fill %d)  prov fired  [range %s]\n",
                                        elapsed, midiToName(pending).c_str(), pending,
                                        midiToName(mpmNote).c_str(), mpmNote,
                                        mpmFill, r.cfg.name.c_str());
                        } else {
                            std::printf("[+%.3fs]  --   MPM confirmed pending OBP→%-4s(%d)"
                                        "  (fill %d)  prov fired  [range %s]\n",
                                        elapsed, midiToName(pending).c_str(), pending,
                                        mpmFill, r.cfg.name.c_str());
                        }
                        std::fflush(stdout);
                        r.provMidiPitch = mpmNote;
                        r.provNote.store(mpmNote, std::memory_order_release);
                        r.monoHeldNote.store(mpmNote, std::memory_order_release);
                    }
                }
                // mpmNote == -1: still not ready, try again next callback
            }
        } else if (gated) {
            resetOBPOnGate(r);
        }

        // Dispatch snapshot: linearise ring → snapshot slot, wake worker.
        dispatchSnapshotIfReady(r, onsetFired, r.provOnTimeMs, m->workerSem, m->gateFloor);
    }

    processSynth(m, out, static_cast<int>(nFrames));
    return 0;
}

// ── main ──────────────────────────────────────────────────────────────────────

int main(int argc, char** argv)
{
    std::string bundlePath, configPath;
    float    threshold       = -1.0f;  // -1 = not set on CLI, use conf/default
    float    frameThreshold  = -1.0f;
    float    gateFloor       = -1.0f;
    float    ampFloor        = -1.0f;
    float    onsetBlankMs    = -1.0f;
    float    swiftThreshold  = -1.0f;
    int      holdCyclesLow   = 2;
    float    windowMs        = 150.0f;
    PlayMode mode            = PlayMode::POLY;
    bool     modeSet         = false;
    Waveform waveform       = Waveform::SINE;
    float    attackMs       = 10.0f;
    float    releaseMs      = 400.0f;
    float    masterVol      = 0.3f;

    for (int i = 1; i < argc; ++i) {
        if      (!std::strcmp(argv[i], "--bundle")          && i+1 < argc) bundlePath     = argv[++i];
        else if (!std::strcmp(argv[i], "--config")          && i+1 < argc) configPath     = argv[++i];
        else if (!std::strcmp(argv[i], "--threshold")       && i+1 < argc) threshold      = std::stof(argv[++i]);
        else if (!std::strcmp(argv[i], "--frame-threshold") && i+1 < argc) frameThreshold = std::stof(argv[++i]);
        else if (!std::strcmp(argv[i], "--hold-cycles")     && i+1 < argc) holdCyclesLow  = std::stoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--gate")            && i+1 < argc) gateFloor      = std::stof(argv[++i]);
        else if (!std::strcmp(argv[i], "--amp-floor")       && i+1 < argc) ampFloor       = std::stof(argv[++i]);
        else if (!std::strcmp(argv[i], "--onset-blank")     && i+1 < argc) onsetBlankMs  = std::stof(argv[++i]);
        else if (!std::strcmp(argv[i], "--swift-threshold") && i+1 < argc) swiftThreshold = std::stof(argv[++i]);
        else if (!std::strcmp(argv[i], "--window")          && i+1 < argc) windowMs       = std::stof(argv[++i]);
        else if (!std::strcmp(argv[i], "--mode")            && i+1 < argc) {
            const char* s = argv[++i];
            if      (!std::strcmp(s, "mono"))      mode = PlayMode::MONO;
            else if (!std::strcmp(s, "swiftmono")) mode = PlayMode::SWIFT_MONO;
            else                                   mode = PlayMode::POLY;
            modeSet = true;
        }
        else if (!std::strcmp(argv[i], "--attack")          && i+1 < argc) attackMs       = std::stof(argv[++i]);
        else if (!std::strcmp(argv[i], "--release")         && i+1 < argc) releaseMs      = std::stof(argv[++i]);
        else if (!std::strcmp(argv[i], "--volume")          && i+1 < argc) masterVol      = std::stof(argv[++i]);
        else if (!std::strcmp(argv[i], "--waveform")        && i+1 < argc) {
            const char* s = argv[++i];
            if      (!std::strcmp(s, "saw"))    waveform = Waveform::SAW;
            else if (!std::strcmp(s, "square")) waveform = Waveform::SQUARE;
            else                                waveform = Waveform::SINE;
        } else {
            std::fprintf(stderr,
                "Usage: %s [--bundle PATH] [--config PATH]\n"
                "          [--threshold F] [--frame-threshold F] [--mode mono|poly|swiftmono]\n"
                "          [--swift-threshold F] [--hold-cycles N] [--gate F] [--amp-floor F]\n"
                "          [--window MS] [--onset-blank MS]\n"
                "          [--waveform sine|saw|square] [--attack MS] [--release MS] [--volume F]\n",
                argv[0]);
            return 1;
        }
    }

    std::string selfDir;
    if (argc > 0) {
        std::string self(argv[0]);
        auto slash = self.rfind('/');
        if (slash != std::string::npos) selfDir = self.substr(0, slash + 1);
    }

    if (bundlePath.empty()) {
        const std::string probes[][2] = {
            { selfDir, selfDir + "ModelData/cnn_contour_model.json" },
            { "/zynthian/zynthian-plugins/lv2/neuralnote_guitar2midi.lv2",
              "/zynthian/zynthian-plugins/lv2/neuralnote_guitar2midi.lv2/ModelData/cnn_contour_model.json" },
            { "/usr/lib/lv2/neuralnote_guitar2midi.lv2",
              "/usr/lib/lv2/neuralnote_guitar2midi.lv2/ModelData/cnn_contour_model.json" },
        };
        for (const auto& p : probes) {
            FILE* f = std::fopen(p[1].c_str(), "rb");
            if (f) { std::fclose(f); bundlePath = p[0]; break; }
        }
        if (bundlePath.empty()) {
            std::fprintf(stderr, "Cannot find model files. Use --bundle <lv2-bundle-dir>\n");
            return 1;
        }
    }

    if (configPath.empty()) {
        std::string c = selfDir + "neuralnote_tune.conf";
        FILE* f = std::fopen(c.c_str(), "r");
        if (f) { std::fclose(f); configPath = c; }
    }

    RangeConfig rangeCfg;
    if (!configPath.empty()) {
        rangeCfg = loadRangeConfig(configPath);
        if (rangeCfg.ranges.empty())
            std::fprintf(stderr, "Warning: config '%s' has no [range] sections.\n", configPath.c_str());
    }

    // CLI overrides (only when explicitly provided)
    if (gateFloor      >= 0.0f)  rangeCfg.gateFloor      = gateFloor;
    if (ampFloor       >= 0.0f)  rangeCfg.ampFloor        = ampFloor;
    if (threshold      >= 0.0f)  rangeCfg.threshold       = threshold;
    if (frameThreshold >= 0.0f)  rangeCfg.frameThreshold  = frameThreshold;
    if (onsetBlankMs   >= 0.0f)  rangeCfg.onsetBlankMs    = onsetBlankMs;
    if (swiftThreshold >= 0.0f)  rangeCfg.swiftF0Threshold = swiftThreshold;
    if (modeSet)                 rangeCfg.mode             = mode;

    if (rangeCfg.ranges.empty()) {
        NoteRange low;
        low.name = "low";  low.midiLow = 0;   low.midiHigh = 48;
        low.windowMs = windowMs; low.holdCycles = holdCyclesLow;
        rangeCfg.ranges.push_back(low);

        NoteRange high;
        high.name = "high"; high.midiLow = 49; high.midiHigh = 127;
        high.windowMs = windowMs; high.holdCycles = 0;
        rangeCfg.ranges.push_back(high);
    }

    std::printf("Bundle:     %s\n", bundlePath.c_str());
    if (!configPath.empty()) std::printf("Config:     %s\n", configPath.c_str());
    std::printf("Gate:       %.4f%s\n", rangeCfg.gateFloor, rangeCfg.gateFloor == 0.0f ? " [disabled]" : "");
    std::printf("AmpFloor:   %.2f\n", rangeCfg.ampFloor);
    std::printf("Threshold:  %.3f\n", rangeCfg.threshold);
    std::printf("FrameThr:   %.3f\n", rangeCfg.frameThreshold);
    std::printf("Mode:       %s\n",
                rangeCfg.mode == PlayMode::MONO       ? "mono" :
                rangeCfg.mode == PlayMode::SWIFT_MONO ? "swiftmono" : "poly");
    if (rangeCfg.mode == PlayMode::SWIFT_MONO)
        std::printf("SwiftThr:   %.2f\n", rangeCfg.swiftF0Threshold);
    std::printf("OnsetBlank: %.0f ms\n", rangeCfg.onsetBlankMs);
    std::printf("Dispatch:   window/2 per range (onset overrides; floor %.0f ms)\n",
                MIN_FRESH_FLOOR / static_cast<float>(PLUGIN_SR) * 1000.0f);
    std::printf("\nNote ranges (%zu, single worker thread):\n", rangeCfg.ranges.size());
    std::printf("  %-12s  %4s  %4s  %6s  %3s  %4s  %s\n",
                "Name","Low","High","WinMs","MNL","Hold","SfHd");
    for (const auto& r : rangeCfg.ranges)
        std::printf("  %-12s  %4d  %4d  %6.0f  %3d  %4d  %d\n",
                    r.name.c_str(), r.midiLow, r.midiHigh,
                    r.windowMs, r.minNoteLength, r.holdCycles,
                    r.swiftHoldCycles);
    const char* wfName = waveform == Waveform::SAW ? "saw" : waveform == Waveform::SQUARE ? "square" : "sine";
    std::printf("\nWaveform:   %s  Attack: %.0f ms  Release: %.0f ms  Volume: %.2f\n\n",
                wfName, attackMs, releaseMs, masterVol);

    try { BinaryData::init(bundlePath); }
    catch (const std::exception& e) { std::fprintf(stderr, "Failed to load models: %s\n", e.what()); return 1; }

    Monitor mon;
    mon.gateFloor       = rangeCfg.gateFloor;
    mon.ampFloor        = rangeCfg.ampFloor;
    mon.threshold       = rangeCfg.threshold;
    mon.frameThreshold  = rangeCfg.frameThreshold;
    mon.onsetBlankMs    = rangeCfg.onsetBlankMs;
    mon.swiftF0Threshold = rangeCfg.swiftF0Threshold;
    mon.mode            = rangeCfg.mode;

    // Try to load SwiftF0 model
    {
        std::string sf0Path;
        // Try next to binary first, then bundle path
        const std::string probes[] = {
            selfDir + "swift_f0_model.onnx",
            bundlePath + (bundlePath.empty() || bundlePath.back() == '/' ? "" : "/") + "swift_f0_model.onnx"
        };
        for (const auto& p : probes) {
            FILE* f = std::fopen(p.c_str(), "rb");
            if (f) { std::fclose(f); sf0Path = p; break; }
        }
        if (!sf0Path.empty()) {
            try {
                mon.swiftF0 = std::make_unique<SwiftF0Detector>(sf0Path);
                std::printf("SwiftF0:    loaded from %s\n", sf0Path.c_str());
            } catch (const std::exception& e) {
                std::fprintf(stderr, "Warning: SwiftF0 model load failed (%s) — "
                             "swiftmono will fall back to BasicPitch\n", e.what());
            }
        } else if (rangeCfg.mode == PlayMode::SWIFT_MONO) {
            std::fprintf(stderr, "Warning: swift_f0_model.onnx not found — "
                         "swiftmono will fall back to BasicPitch\n");
        }
    }
    mon.waveform  = waveform;
    mon.attackMs  = attackMs;
    mon.releaseMs = releaseMs;
    mon.masterVol = masterVol;
    mon.startTime = std::chrono::steady_clock::now();
    g_mon         = &mon;
    sem_init(&mon.workerSem, 0, 0);

    for (const auto& rc : rangeCfg.ranges) {
        auto r             = std::make_unique<RangeState>();
        r->cfg             = rc;
        r->ringSize        = windowMsToRingSize(rc.windowMs);
        r->minFreshSamples = std::max(r->ringSize / 2, MIN_FRESH_FLOOR);
        r->ring.assign(RING_MAX, 0.0f);
        r->basicPitch      = std::make_unique<BasicPitch>();
        mon.ranges.push_back(std::move(r));
    }

    jack_status_t  status;
    jack_client_t* client = jack_client_open("neuralnote_tune", JackNullOption, &status);
    if (!client) { std::fprintf(stderr, "JACK connection failed (0x%x)\n", status); return 1; }

    mon.sampleRate = jack_get_sample_rate(client);
    std::printf("JACK:       %.0f Hz, buffer %u frames\n", mon.sampleRate, jack_get_buffer_size(client));

    // Configure HPF pick detector.
    mon.pickDetector.init(static_cast<float>(mon.sampleRate));

    // Configure per-range OBP lowpass and MPM (both need sample rate).
    for (auto& rp : mon.ranges) {
        const float sr     = static_cast<float>(mon.sampleRate);
        const float cutoff = std::min(midiToFreq(rp->cfg.midiHigh) * 1.2f, sr * 0.45f);
        rp->obd.setLowpass(cutoff, sr);
        rp->mpm.init(sr, rp->cfg.midiLow, rp->cfg.midiHigh);
    }

    mon.inPort  = jack_port_register(client, "audio_in",  JACK_DEFAULT_AUDIO_TYPE, JackPortIsInput,  0);
    mon.outPort = jack_port_register(client, "audio_out", JACK_DEFAULT_AUDIO_TYPE, JackPortIsOutput, 0);
    if (!mon.inPort || !mon.outPort) {
        std::fprintf(stderr, "Failed to register JACK ports\n");
        jack_client_close(client); return 1;
    }

    JackCtx ctx{ &mon };
    jack_set_process_callback(client, jackProcess, &ctx);

    mon.workerThread = std::thread(runWorker, &mon);

    if (jack_activate(client) != 0) {
        std::fprintf(stderr, "Cannot activate JACK client\n");
        mon.workerQuit.store(true, std::memory_order_release);
        sem_post(&mon.workerSem);
        mon.workerThread.join();
        jack_client_close(client); return 1;
    }

    const char** caps = jack_get_ports(client, nullptr, JACK_DEFAULT_AUDIO_TYPE, JackPortIsPhysical | JackPortIsOutput);
    if (caps && caps[0]) { jack_connect(client, caps[0], jack_port_name(mon.inPort)); std::printf("Audio in:   %s\n", caps[0]); }
    if (caps) jack_free(caps);

    const char** plays = jack_get_ports(client, nullptr, JACK_DEFAULT_AUDIO_TYPE, JackPortIsPhysical | JackPortIsInput);
    if (plays) {
        if (plays[0]) { jack_connect(client, jack_port_name(mon.outPort), plays[0]); std::printf("Audio out:  %s", plays[0]); }
        if (plays[1]) { jack_connect(client, jack_port_name(mon.outPort), plays[1]); std::printf(", %s", plays[1]); }
        if (plays[0]) std::printf("\n");
        jack_free(plays);
    }

    std::printf("\nListening. Ctrl+C to stop.\n");
    std::printf("%-12s  %-5s  %-6s  %5s  %3s  %s\n", "Time","Event","Note","MIDI#","Vel","Info");
    std::printf("%-12s  %-5s  %-6s  %5s  %3s  %s\n", "------------","-----","------","-----","---","------------------------");

    std::signal(SIGINT,  onSignal);
    std::signal(SIGTERM, onSignal);
    while (!g_quit.load()) std::this_thread::sleep_for(std::chrono::milliseconds(50));

    std::printf("\nShutting down...\n");
    jack_deactivate(client);
    jack_client_close(client);
    mon.workerQuit.store(true, std::memory_order_release);
    sem_post(&mon.workerSem);
    if (mon.workerThread.joinable()) mon.workerThread.join();
    sem_destroy(&mon.workerSem);
    return 0;
}
