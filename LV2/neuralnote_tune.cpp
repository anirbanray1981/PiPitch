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
 *                   [--threshold 0.6] [--frame-threshold 0.5]
 *                   [--min-note-length 6] [--hold-cycles 2]
 *                   [--gate 0.003] [--amp-floor 0.65] [--window 150]
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

#include "BinaryData.h"
#include "BasicPitch.h"
#include "BasicPitchConstants.h"
#include "NoteRangeConfig.h"
#include "OneBitPitchDetector.h"

// ── Constants ──────────────────────────────────────────────────────────────────

static constexpr double PLUGIN_SR        = 22050.0;
static constexpr int    RING_MAX         = static_cast<int>(PLUGIN_SR * 2.0);
static constexpr int    MAX_VOICES       = 16;
static constexpr int    MIN_FRESH_FLOOR  = static_cast<int>(PLUGIN_SR * 0.025);
static constexpr int    MIDI_QUEUE_CAP   = 64;

static constexpr float  ONSET_RATIO    = 3.0f;
static constexpr float  ONSET_ALPHA    = 0.05f;
static constexpr float  ONSET_BLANK_MS = 50.0f;

static int windowMsToRingSize(float ms)
{
    const float c = std::clamp(ms, 35.0f, 2000.0f);
    return std::min(static_cast<int>(c / 1000.0f * PLUGIN_SR), RING_MAX);
}

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

struct PendingNote { bool noteOn; int pitch; float velocity; };

// ── Lockless SPSC snapshot channel (jackProcess → worker) ─────────────────────

struct SnapshotChannel {
    std::vector<float> data;
    int                snapshotSize       = 0;
    int                provNoteAtDispatch = -1;
    double             provOnMs           = 0.0;  // ms since startTime when provisional fired
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

// ── Lockless SPSC MIDI output queue (worker → jackProcess) ────────────────────

struct MidiOutQueue {
    PendingNote      buf[MIDI_QUEUE_CAP];
    std::atomic<int> head{0};
    std::atomic<int> tail{0};

    void push(PendingNote n) {
        const int t    = tail.load(std::memory_order_relaxed);
        const int next = (t + 1) % MIDI_QUEUE_CAP;
        if (next == head.load(std::memory_order_acquire)) return;
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

// ── Per-range runtime state ────────────────────────────────────────────────────

struct RangeState {
    NoteRange cfg;

    std::unique_ptr<BasicPitch> bp;

    std::vector<float> ring;
    int ringHead         = 0;
    int ringFilled       = 0;
    int freshSamples     = 0;
    int ringSize         = 0;
    int minFreshSamples  = 0;  // = max(ringSize/2, MIN_FRESH_FLOOR)

    SnapshotChannel snapChan;
    MidiOutQueue    midiOut;

    std::set<int>      activeNotes;
    std::map<int, int> noteHold;

    OneBitPitchDetector obd;
    std::atomic<int>    provNote{-1};   // set by jackProcess, cleared by worker
    int                 provMidiPitch = -1;  // within-callback: consumed by processSynth
    double              provOnTimeMs  = 0.0; // set by processSynth, copied to snapChan at dispatch
    double              obdMinHoldMs  = 0.0; // min ms before CNN may cancel an OBP provisional

    // OBP onset gate and voting buffer
    bool            obdOnsetActive  = false;
    int             obdWindowRemain = 0;    // samples remaining in OBP window; expires after 100 ms
    OBPVotingBuffer obdVoting;

    std::thread workerThread;

    RangeState()                           = default;
    RangeState(const RangeState&)          = delete;
    RangeState& operator=(const RangeState&) = delete;
    RangeState(RangeState&&)               = delete;
    RangeState& operator=(RangeState&&)    = delete;
};

// ── Shared state ───────────────────────────────────────────────────────────────

struct Monitor {
    double sampleRate = 48000.0;
    float  gateFloor  = 0.003f;
    float  ampFloor   = 0.65f;

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
    int   onsetBlankRemain = 0;      // raw input samples remaining in blank period

    // Per-range OBP provisional note fired this onset (-1 if none); cleared on each onset.
    // Used for cross-range harmonic suppression.
    std::array<int, 8> onsetProvNotes = {-1,-1,-1,-1,-1,-1,-1,-1};
};

static Monitor*          g_mon = nullptr;
static std::atomic<bool> g_quit{false};
static void onSignal(int) { g_quit.store(true); }

// ── Worker thread (one per range) ──────────────────────────────────────────────

static void applyRangeDiff(Monitor* m, RangeState& r,
                           const std::map<int, int>& newNotesCNN, double inferMs, int prov)
{
    const double elapsed   = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - m->startTime).count();
    const double elapsedMs = elapsed * 1000.0;

    // If CNN wants to cancel a provisional but the minimum hold hasn't elapsed,
    // keep the provisional alive by treating it as confirmed for this pass.
    // This prevents double-fire glitches when onset-forced dispatch runs on stale audio.
    const bool holdElapsed = (elapsedMs - r.snapChan.provOnMs) >= r.obdMinHoldMs;
    const std::map<int, int>* pNewNotes = &newNotesCNN;
    std::map<int, int> heldNotes;
    if (prov != -1 && newNotesCNN.empty() && !holdElapsed) {
        heldNotes[prov] = 80;   // keep provisional as if confirmed
        pNewNotes = &heldNotes;
    }
    const std::map<int, int>& newNotes = *pNewNotes;

    // Log CNN outcome relative to any provisional OBP note
    if (prov != -1) {
        if (newNotes.count(prov)) {
            if (pNewNotes == &heldNotes) {
                std::printf("[+%.3fs]  --   OBP hold  %-4s (%3d)"
                            "  [%.0f/%.0f ms  range %s]\n",
                            elapsed, midiToName(prov).c_str(), prov,
                            elapsedMs - r.snapChan.provOnMs, r.obdMinHoldMs,
                            r.cfg.name.c_str());
            } else {
                std::printf("[+%.3fs]  --   CNN confirmed OBP %-4s (%3d)"
                            "  [inf %4.0fms  range %s]\n",
                            elapsed, midiToName(prov).c_str(), prov,
                            inferMs, r.cfg.name.c_str());
            }
        } else if (!newNotes.empty()) {
            std::printf("[+%.3fs]  --   CNN corrected OBP %-4s (%3d) →",
                        elapsed, midiToName(prov).c_str(), prov);
            for (const auto& [p, v] : newNotes)
                std::printf("  %-4s (%3d)", midiToName(p).c_str(), p);
            std::printf("  [inf %4.0fms  range %s]\n", inferMs, r.cfg.name.c_str());
        } else {
            std::printf("[+%.3fs]  --   CNN cancelled OBP %-4s (%3d)"
                        "  [inf %4.0fms  range %s]\n",
                        elapsed, midiToName(prov).c_str(), prov,
                        inferMs, r.cfg.name.c_str());
        }
        std::fflush(stdout);
    }

    for (const auto& [p, v] : newNotes) {
        r.noteHold.erase(p);
        if (!r.activeNotes.count(p)) {
            std::printf("[+%.3fs]  ON   %-4s (%3d)  vel %3d"
                        "  [CNN  win %4.0fms  inf %4.0fms  range %s]\n",
                        elapsed, midiToName(p).c_str(), p, v,
                        r.cfg.windowMs, inferMs, r.cfg.name.c_str());
            std::fflush(stdout);
            r.midiOut.push({true, p, v / 127.0f});
            r.activeNotes.insert(p);
        }
    }

    for (auto it = r.noteHold.begin(); it != r.noteHold.end(); ) {
        if (--(it->second) <= 0) {
            const int p = it->first;
            std::printf("[+%.3fs]  OFF  %-4s (%3d)\n", elapsed, midiToName(p).c_str(), p);
            std::fflush(stdout);
            r.midiOut.push({false, p, 0.0f});
            r.activeNotes.erase(p);
            it = r.noteHold.erase(it);
        } else { ++it; }
    }

    std::vector<int> immediateOff;
    for (int p : r.activeNotes) {
        if (newNotes.count(p) || r.noteHold.count(p)) continue;
        if (r.cfg.holdCycles > 0) {
            r.noteHold[p] = r.cfg.holdCycles;
        } else {
            std::printf("[+%.3fs]  OFF  %-4s (%3d)\n", elapsed, midiToName(p).c_str(), p);
            std::fflush(stdout);
            r.midiOut.push({false, p, 0.0f});
            immediateOff.push_back(p);
        }
    }
    for (int p : immediateOff) r.activeNotes.erase(p);
}

static void runWorkerForRange(Monitor* m, RangeState* r)
{
    while (true) {
        sem_wait(&r->snapChan.sem);

        if (r->snapChan.quit.load(std::memory_order_acquire)) break;

        const float minDurMs = r->cfg.minNoteLength * FFT_HOP / static_cast<float>(PLUGIN_SR) * 1000.0f;
        r->bp->setParameters(1.0f - r->cfg.frameThreshold, r->cfg.threshold, minDurMs);

        const auto t0 = std::chrono::steady_clock::now();
        r->bp->transcribeToMIDI(r->snapChan.data.data(), r->snapChan.snapshotSize);
        r->snapChan.ready.store(false, std::memory_order_release);
        const double inferMs = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - t0).count();

        // Two-phase: insert provisional OneBitPitch note before diff
        const int prov = r->snapChan.provNoteAtDispatch;
        if (prov != -1) {
            r->activeNotes.insert(prov);
            r->snapChan.provNoteAtDispatch = -1;
            r->provNote.store(-1, std::memory_order_release);
        }

        std::map<int, int> newNotes;
        for (const auto& ev : r->bp->getNoteEvents()) {
            if (static_cast<float>(ev.amplitude) < m->ampFloor) continue;
            const int p = static_cast<int>(ev.pitch);
            if (p < r->cfg.midiLow || p > r->cfg.midiHigh) continue;
            const int v = std::clamp(static_cast<int>(ev.amplitude * 127.0), 1, 127);
            auto it = newNotes.find(p);
            if (it == newNotes.end() || v > it->second) newNotes[p] = v;
        }

        applyRangeDiff(m, *r, newNotes, inferMs, prov);
    }

    // Shutdown: release all active notes
    const double elapsed = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - m->startTime).count();
    for (int p : r->activeNotes) {
        std::printf("[+%.3fs]  OFF  %-4s (%3d)  [shutdown]\n",
                    elapsed, midiToName(p).c_str(), p);
        r->midiOut.push({false, p, 0.0f});
    }
    r->activeNotes.clear();
    r->noteHold.clear();
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

        // Check if a lower range already fired something this onset that makes this one
        // a harmonic (diff 12 or 24) or an adjacent-frequency artefact (diff ≤ 1).
        bool suppressed = false;
        for (int j = 0; j < ri && !suppressed; ++j) {
            const int pj = m->onsetProvNotes[j];
            if (pj == -1) continue;
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
                int   best      = 0;
                float bestLevel = m->voices[0].envLevel + (m->voices[0].state != 0 ? 1.0f : 0.0f);
                for (int i = 0; i < MAX_VOICES; ++i) {
                    if (m->voices[i].state == 0) { best = i; break; }
                    const float l = m->voices[i].envLevel;
                    if (l < bestLevel) { bestLevel = l; best = i; }
                }
                m->voices[best] = { pn.pitch, midiToFreq(pn.pitch), 0.0, pn.velocity, 1, 0.0f };
            } else {
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

    // Onset detection: force-dispatch all ready rings when a pick attack is detected.
    bool onsetFired = false;
    if (m->onsetBlankRemain > 0) {
        m->onsetBlankRemain -= static_cast<int>(nFrames);
        if (m->onsetBlankRemain < 0) m->onsetBlankRemain = 0;
    } else if (!gated && blockRms > m->onsetSmoothedRms * ONSET_RATIO) {
        onsetFired            = true;
        m->onsetBlankRemain   = static_cast<int>(m->sampleRate * (ONSET_BLANK_MS / 1000.0f));
        m->onsetSmoothedRms   = blockRms;
        m->onsetProvNotes.fill(-1);  // new onset: reset cross-range provisional tracking
    }
    if (!onsetFired && m->onsetBlankRemain == 0)
        m->onsetSmoothedRms = m->onsetSmoothedRms * (1.0f - ONSET_ALPHA) + blockRms * ONSET_ALPHA;

    // Resample once, shared across all ranges
    std::vector<float> resampled;
    resampled.reserve(static_cast<int>(nFrames * (PLUGIN_SR / m->sampleRate)) + 2);
    if (gated) resampled.assign(static_cast<int>(nFrames * (PLUGIN_SR / m->sampleRate)), 0.0f);
    else       resampleLinear(in, static_cast<int>(nFrames), m->sampleRate, resampled);

    for (auto& rp : m->ranges) {
        RangeState& r  = *rp;
        const int   rs = r.ringSize;

        for (float s : resampled) {
            r.ring[r.ringHead] = s;
            r.ringHead = (r.ringHead + 1) % rs;
            if (r.ringFilled < rs) ++r.ringFilled;
        }
        r.freshSamples += static_cast<int>(resampled.size());

        // Two-phase OneBitPitch: arm on onset, expire after 100 ms.
        // Voting buffer requires 9/12 consistent readings before firing provisional.
        if (onsetFired) {
            r.obdOnsetActive  = true;
            r.obdWindowRemain = static_cast<int>(m->sampleRate * 0.1f);
            r.obdVoting.reset();
            r.obd.reset();
        } else if (r.obdWindowRemain > 0) {
            r.obdWindowRemain -= static_cast<int>(nFrames);
            if (r.obdWindowRemain <= 0) {
                r.obdWindowRemain = 0;
                r.obdOnsetActive  = false;  // window expired — give up
            }
        }
        if (r.obdOnsetActive && !gated && r.provNote.load(std::memory_order_relaxed) == -1) {
            int op = r.obd.process(in, static_cast<int>(nFrames),
                                   static_cast<float>(m->sampleRate));
            // Feed OBP output into voting buffer; only fire when supermajority reached.
            const int voted = r.obdVoting.update(
                (op >= r.cfg.midiLow && op <= r.cfg.midiHigh) ? op : -1);
            if (voted != -1) {
                // Suppress if any other range already has a live provisional
                // that is the fundamental of this note (voted - 12 or - 24).
                // That means we are detecting a harmonic, not the real pitch.
                bool isHarmonic = false;
                for (const auto& other : m->ranges) {
                    if (other.get() == &r) continue;
                    const int op = other->provNote.load(std::memory_order_relaxed);
                    if (op != -1) {
                        const int diff = voted - op;
                        if (diff == 12 || diff == 24) { isHarmonic = true; break; }
                    }
                }
                // Arm off regardless; if harmonic, don't fire — just drop this onset.
                r.obdOnsetActive  = false;
                r.obdWindowRemain = 0;
                r.obdVoting.reset();
                if (!isHarmonic) {
                    r.provMidiPitch = voted;
                    r.provNote.store(voted, std::memory_order_release);
                }
            }
        } else if (gated) {
            r.obd.reset();
            r.obdVoting.reset();
            r.obdOnsetActive  = false;
            r.obdWindowRemain = 0;
        }

        // Dispatch snapshot locklessly: normal path (ringSize/2 fresh audio) or onset path.
        if (!r.snapChan.ready.load(std::memory_order_acquire)
            && r.ringFilled >= rs
            && (r.freshSamples >= r.minFreshSamples || onsetFired))
        {
            const int tail = r.ringHead;
            const int p1   = rs - tail;
            std::memcpy(r.snapChan.data.data(),       r.ring.data() + tail, p1 * sizeof(float));
            if (p1 < rs)
                std::memcpy(r.snapChan.data.data() + p1, r.ring.data(), (rs - p1) * sizeof(float));
            r.snapChan.snapshotSize       = rs;
            r.snapChan.provNoteAtDispatch = r.provNote.load(std::memory_order_relaxed);
            r.snapChan.provOnMs           = r.provOnTimeMs;
            r.snapChan.ready.store(true, std::memory_order_release);
            r.freshSamples = 0;
            sem_post(&r.snapChan.sem);
        }
    }

    processSynth(m, out, static_cast<int>(nFrames));
    return 0;
}

// ── main ──────────────────────────────────────────────────────────────────────

int main(int argc, char** argv)
{
    std::string bundlePath, configPath;
    float    threshold      = 0.6f;
    float    frameThreshold = 0.5f;
    int      minNoteLength  = 6;
    float    gateFloor      = 0.003f;
    int      holdCyclesLow  = 2;
    float    ampFloor       = 0.65f;
    float    windowMs       = 150.0f;
    Waveform waveform       = Waveform::SINE;
    float    attackMs       = 10.0f;
    float    releaseMs      = 400.0f;
    float    masterVol      = 0.3f;

    for (int i = 1; i < argc; ++i) {
        if      (!std::strcmp(argv[i], "--bundle")          && i+1 < argc) bundlePath     = argv[++i];
        else if (!std::strcmp(argv[i], "--config")          && i+1 < argc) configPath     = argv[++i];
        else if (!std::strcmp(argv[i], "--threshold")       && i+1 < argc) threshold      = std::stof(argv[++i]);
        else if (!std::strcmp(argv[i], "--frame-threshold") && i+1 < argc) frameThreshold = std::stof(argv[++i]);
        else if (!std::strcmp(argv[i], "--min-note-length") && i+1 < argc) minNoteLength  = std::stoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--hold-cycles")     && i+1 < argc) holdCyclesLow  = std::stoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--gate")            && i+1 < argc) gateFloor      = std::stof(argv[++i]);
        else if (!std::strcmp(argv[i], "--amp-floor")       && i+1 < argc) ampFloor       = std::stof(argv[++i]);
        else if (!std::strcmp(argv[i], "--window")          && i+1 < argc) windowMs       = std::stof(argv[++i]);
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
                "          [--threshold F] [--frame-threshold F] [--min-note-length N]\n"
                "          [--hold-cycles N] [--gate F] [--amp-floor F] [--window MS]\n"
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
        rangeCfg.gateFloor = gateFloor;
        rangeCfg.ampFloor  = ampFloor;
    }

    if (rangeCfg.ranges.empty()) {
        NoteRange low;
        low.name = "low";   low.midiLow = 0;   low.midiHigh = 48;
        low.windowMs = windowMs; low.threshold = threshold;
        low.frameThreshold = frameThreshold; low.minNoteLength = minNoteLength;
        low.holdCycles = holdCyclesLow;
        rangeCfg.ranges.push_back(low);

        NoteRange high;
        high.name = "high"; high.midiLow = 49; high.midiHigh = 127;
        high.windowMs = windowMs; high.threshold = threshold;
        high.frameThreshold = frameThreshold; high.minNoteLength = minNoteLength;
        high.holdCycles = 0;
        rangeCfg.ranges.push_back(high);

        rangeCfg.gateFloor = gateFloor;
        rangeCfg.ampFloor  = ampFloor;
    }

    std::printf("Bundle:     %s\n", bundlePath.c_str());
    if (!configPath.empty()) std::printf("Config:     %s\n", configPath.c_str());
    std::printf("Gate:       %.4f%s\n", rangeCfg.gateFloor, rangeCfg.gateFloor == 0.0f ? " [disabled]" : "");
    std::printf("AmpFloor:   %.2f\n", rangeCfg.ampFloor);
    std::printf("Dispatch:   window/2 per range (onset overrides; floor %.0f ms)\n",
                MIN_FRESH_FLOOR / static_cast<float>(PLUGIN_SR) * 1000.0f);
    std::printf("\nNote ranges (%zu, one parallel worker thread each):\n", rangeCfg.ranges.size());
    std::printf("  %-12s  %4s  %4s  %6s  %5s  %5s  %3s  %s\n",
                "Name","Low","High","WinMs","Thr","FrThr","MNL","Hold");
    for (const auto& r : rangeCfg.ranges)
        std::printf("  %-12s  %4d  %4d  %6.0f  %.3f  %.3f  %3d  %d\n",
                    r.name.c_str(), r.midiLow, r.midiHigh,
                    r.windowMs, r.threshold, r.frameThreshold, r.minNoteLength, r.holdCycles);
    const char* wfName = waveform == Waveform::SAW ? "saw" : waveform == Waveform::SQUARE ? "square" : "sine";
    std::printf("\nWaveform:   %s  Attack: %.0f ms  Release: %.0f ms  Volume: %.2f\n\n",
                wfName, attackMs, releaseMs, masterVol);

    try { BinaryData::init(bundlePath); }
    catch (const std::exception& e) { std::fprintf(stderr, "Failed to load models: %s\n", e.what()); return 1; }

    Monitor mon;
    mon.gateFloor = rangeCfg.gateFloor;
    mon.ampFloor  = rangeCfg.ampFloor;
    mon.waveform  = waveform;
    mon.attackMs  = attackMs;
    mon.releaseMs = releaseMs;
    mon.masterVol = masterVol;
    mon.startTime = std::chrono::steady_clock::now();
    g_mon         = &mon;

    static constexpr float CNN_FRAME_MS = 11.6f;  // 1 CNN output frame ≈ 11.6 ms

    for (const auto& rc : rangeCfg.ranges) {
        auto r             = std::make_unique<RangeState>();
        r->cfg             = rc;
        r->ringSize        = windowMsToRingSize(rc.windowMs);
        r->minFreshSamples = std::max(r->ringSize / 2, MIN_FRESH_FLOOR);
        r->ring.assign(RING_MAX, 0.0f);
        r->bp              = std::make_unique<BasicPitch>();
        r->obdMinHoldMs    = 5.0 * rc.minNoteLength * CNN_FRAME_MS;
        mon.ranges.push_back(std::move(r));
    }

    jack_status_t  status;
    jack_client_t* client = jack_client_open("neuralnote_tune", JackNullOption, &status);
    if (!client) { std::fprintf(stderr, "JACK connection failed (0x%x)\n", status); return 1; }

    mon.sampleRate = jack_get_sample_rate(client);
    std::printf("JACK:       %.0f Hz, buffer %u frames\n", mon.sampleRate, jack_get_buffer_size(client));

    // Configure per-range OBP lowpass: cutoff at 1.5× range max freq, capped below Nyquist.
    for (auto& rp : mon.ranges) {
        const float cutoff = std::min(midiToFreq(rp->cfg.midiHigh) * 1.5f,
                                      static_cast<float>(mon.sampleRate) * 0.45f);
        rp->obd.setLowpass(cutoff, static_cast<float>(mon.sampleRate));
    }

    mon.inPort  = jack_port_register(client, "audio_in",  JACK_DEFAULT_AUDIO_TYPE, JackPortIsInput,  0);
    mon.outPort = jack_port_register(client, "audio_out", JACK_DEFAULT_AUDIO_TYPE, JackPortIsOutput, 0);
    if (!mon.inPort || !mon.outPort) {
        std::fprintf(stderr, "Failed to register JACK ports\n");
        jack_client_close(client); return 1;
    }

    JackCtx ctx{ &mon };
    jack_set_process_callback(client, jackProcess, &ctx);

    for (auto& rp : mon.ranges)
        rp->workerThread = std::thread(runWorkerForRange, &mon, rp.get());

    if (jack_activate(client) != 0) {
        std::fprintf(stderr, "Cannot activate JACK client\n");
        for (auto& rp : mon.ranges) {
            rp->snapChan.quit.store(true, std::memory_order_release);
            sem_post(&rp->snapChan.sem);
            rp->workerThread.join();
        }
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
    for (auto& rp : mon.ranges) {
        rp->snapChan.quit.store(true, std::memory_order_release);
        sem_post(&rp->snapChan.sem);
    }
    for (auto& rp : mon.ranges)
        if (rp->workerThread.joinable()) rp->workerThread.join();
    return 0;
}
