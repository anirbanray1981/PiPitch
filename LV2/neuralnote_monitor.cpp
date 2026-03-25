/**
 * neuralnote_monitor — NeuralNote terminal note monitor
 *
 * Connects to JACK, captures audio from system:capture_1, runs BasicPitch
 * inference in a background thread, and prints detected notes to stdout.
 *
 * Usage:
 *   neuralnote_monitor [--bundle PATH] [--threshold FLOAT] [--gate FLOAT] [--mode 0|1|2]
 *
 *   --bundle PATH      Directory containing ModelData/  (default: same dir as binary,
 *                      then /zynthian/zynthian-plugins/lv2/neuralnote_guitar2midi.lv2)
 *   --threshold FLOAT  Note sensitivity 0.1-1.0  (default 0.5; lower = fewer/cleaner notes)
 *   --gate FLOAT       Noise gate floor as linear RMS 0.0-0.1  (default 0.003 ≈ -50 dBFS)
 *                      Blocks below this level are treated as silence; set 0 to disable
 *   --min-dur MS       Minimum note duration in ms  (default 100; higher = fewer harmonics)
 *   --amp-floor FLOAT  Minimum note amplitude 0.0-1.0  (default 0.65; filters weak artifacts)
 *   --window MS        Inference window 50-2000 ms  (default 300)
 *
 * Output format (one line per event):
 *   [+0.123s]  NOTE ON   A4  (69)
 *   [+0.456s]  NOTE OFF  A4  (69)
 */

#include <algorithm>
#include <atomic>
#include <map>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <mutex>
#include <set>
#include <string>
#include <thread>
#include <vector>
#include <csignal>

#include <jack/jack.h>

// BinaryData.h must come before any Lib/Model header
#include "BinaryData.h"
#include "BasicPitch.h"
#include "BasicPitchConstants.h"

// ── Constants ─────────────────────────────────────────────────────────────────

static constexpr double PLUGIN_SR = 22050.0;           // BasicPitch native rate
static constexpr int    RING_MAX  = static_cast<int>(PLUGIN_SR * 2.0); // 2 s max

static int windowMsToRingSize(float ms)
{
    const float clamped = std::clamp(ms, 50.0f, 2000.0f);
    return std::min(static_cast<int>(clamped / 1000.0f * PLUGIN_SR), RING_MAX);
}

// ── Note name helpers ─────────────────────────────────────────────────────────

static const char* NOTE_NAMES[12] = {
    "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"
};

static std::string midiToName(int midi)
{
    char buf[16];
    int  octave = (midi / 12) - 1;
    std::snprintf(buf, sizeof(buf), "%s%d", NOTE_NAMES[midi % 12], octave);
    return buf;
}

// ── Linear resampler (matches neuralnote_impl.cpp) ───────────────────────────

static void resampleLinear(const float* in, int inLen, double srcRate,
                            std::vector<float>& out)
{
    if (srcRate == PLUGIN_SR) {
        out.insert(out.end(), in, in + inLen);
        return;
    }
    const double ratio  = PLUGIN_SR / srcRate;
    const int    outLen = static_cast<int>(inLen * ratio);
    for (int i = 0; i < outLen; ++i) {
        const double srcPos = i / ratio;
        const int    s0     = static_cast<int>(srcPos);
        const double frac   = srcPos - s0;
        const int    s1     = std::min(s0 + 1, inLen - 1);
        out.push_back(static_cast<float>((1.0 - frac) * in[s0] + frac * in[s1]));
    }
}

// ── Shared state ──────────────────────────────────────────────────────────────

struct Monitor {
    // --- config (set once before threads start) ---
    double   sampleRate   = 48000.0;
    float    threshold    = 0.5f;    // note sensitivity (lower = fewer, less harmonics)
    float    gateFloor    = 0.003f;  // RMS below this → silence (≈ -50 dBFS)
    float    minDurMs     = 100.0f;  // min note duration in ms (filters brief harmonic flickers)
    float    ampFloor     = 0.65f;   // min note amplitude 0.0-1.0 (filters weak harmonic artifacts)
    int      ringSize     = windowMsToRingSize(300.0f);

    // --- ring buffer (written by JACK process callback) ---
    float    ring[RING_MAX] = {};
    int      ringHead   = 0;
    int      ringFilled = 0;

    // --- worker thread ---
    std::thread             workerThread;
    std::mutex              workerMutex;
    std::condition_variable workerCv;
    bool                    workerHasWork = false;
    bool                    workerQuit    = false;
    std::vector<float>      workerSnapshot;

    // --- active notes (worker only) ---
    std::set<int> activeNotes;

    // --- start time for relative timestamps ---
    std::chrono::steady_clock::time_point startTime;

    // --- inference engine (worker only) ---
    std::unique_ptr<BasicPitch> bp;
};

static Monitor* g_mon = nullptr;
static std::atomic<bool> g_quit{false};

// ── Signal handler ────────────────────────────────────────────────────────────

static void onSignal(int)
{
    g_quit.store(true);
}

// ── Worker thread ─────────────────────────────────────────────────────────────

static void runWorker(Monitor* m)
{
    while (true) {
        std::vector<float> snap;
        {
            std::unique_lock<std::mutex> lk(m->workerMutex);
            m->workerCv.wait(lk, [m]{ return m->workerHasWork || m->workerQuit; });
            if (m->workerQuit) break;
            m->workerHasWork = false;
            snap = m->workerSnapshot;
        }

        m->bp->setParameters(m->threshold, 0.5f, m->minDurMs);
        m->bp->transcribeToMIDI(snap.data(), static_cast<int>(snap.size()));

        // Collect pitches from this inference window, filtering by amplitude
        std::map<int, int> newNotes; // pitch → velocity (1-127)
        for (const auto& ev : m->bp->getNoteEvents()) {
            if (static_cast<float>(ev.amplitude) >= m->ampFloor) {
                const int p = static_cast<int>(ev.pitch);
                const int v = std::clamp(static_cast<int>(ev.amplitude * 127.0), 1, 127);
                auto it = newNotes.find(p);
                if (it == newNotes.end() || v > it->second)
                    newNotes[p] = v;
            }
        }

        // Compute elapsed time for log prefix
        const double elapsed =
            std::chrono::duration<double>(
                std::chrono::steady_clock::now() - m->startTime).count();

        // Diff against active set → print events
        for (const auto& [p, v] : newNotes) {
            if (m->activeNotes.find(p) == m->activeNotes.end()) {
                std::printf("[+%.3fs]  NOTE ON   %-4s (%3d)  vel %3d\n",
                            elapsed, midiToName(p).c_str(), p, v);
                std::fflush(stdout);
            }
        }
        for (int p : m->activeNotes) {
            if (newNotes.find(p) == newNotes.end()) {
                std::printf("[+%.3fs]  NOTE OFF  %-4s (%3d)\n",
                            elapsed, midiToName(p).c_str(), p);
                std::fflush(stdout);
            }
        }
        // rebuild active set from new pitches
        m->activeNotes.clear();
        for (const auto& [p, v] : newNotes)
            m->activeNotes.insert(p);
    }

    // All notes off on exit
    const double elapsed =
        std::chrono::duration<double>(
            std::chrono::steady_clock::now() - m->startTime).count();
    for (int p : m->activeNotes) {
        std::printf("[+%.3fs]  NOTE OFF  %-4s (%3d)  [shutdown]\n",
                    elapsed, midiToName(p).c_str(), p);
    }
    m->activeNotes.clear();
}

// ── JACK context and process callback ────────────────────────────────────────

struct JackCtx {
    jack_port_t* port;
    Monitor*     mon;
};

static int jackProcessImpl(jack_nframes_t nFrames, void* arg)
{
    JackCtx* ctx = static_cast<JackCtx*>(arg);
    Monitor* m   = ctx->mon;

    const float* in = static_cast<const float*>(
        jack_port_get_buffer(ctx->port, nFrames));

    // Noise gate: compute RMS of this block; if below floor use silence
    float sumSq = 0.0f;
    for (jack_nframes_t i = 0; i < nFrames; ++i)
        sumSq += in[i] * in[i];
    const float rms = std::sqrt(sumSq / static_cast<float>(nFrames));
    const bool  gated = (rms < m->gateFloor);

    // Resample into a temporary vector then push to ring
    std::vector<float> resampled;
    resampled.reserve(static_cast<int>(nFrames * (PLUGIN_SR / m->sampleRate)) + 2);
    if (gated) {
        // Push zeros so old audio flushes out of the ring
        const int outLen = static_cast<int>(nFrames * (PLUGIN_SR / m->sampleRate));
        resampled.assign(outLen, 0.0f);
    } else {
        resampleLinear(in, static_cast<int>(nFrames), m->sampleRate, resampled);
    }

    // Write resampled frames into the ring (overwrite oldest on overflow)
    const int rs = m->ringSize;
    for (float s : resampled) {
        m->ring[m->ringHead] = s;
        m->ringHead = (m->ringHead + 1) % rs;
        if (m->ringFilled < rs) ++m->ringFilled;
    }

    // If ring is full, hand a linearised snapshot to the worker (non-blocking)
    if (m->ringFilled >= rs) {
        std::unique_lock<std::mutex> lk(m->workerMutex, std::try_to_lock);
        if (lk.owns_lock() && !m->workerHasWork) {
            m->workerSnapshot.resize(rs);
            // tail = index of oldest sample in the ring
            const int tail  = m->ringHead; // head points to oldest (next write pos)
            const int part1 = rs - tail;   // samples from tail to end of used region
            std::memcpy(m->workerSnapshot.data(),
                        m->ring + tail, part1 * sizeof(float));
            if (part1 < rs)
                std::memcpy(m->workerSnapshot.data() + part1,
                            m->ring, (rs - part1) * sizeof(float));
            m->workerHasWork = true;
            m->workerCv.notify_one();
        }
    }

    return 0;
}

// ── main ──────────────────────────────────────────────────────────────────────

int main(int argc, char** argv)
{
    // --- parse args ---
    std::string bundlePath;
    float threshold = 0.5f;     // note sensitivity (lower = fewer notes, less harmonics)
    float gateFloor = 0.003f;   // RMS below this → silence (≈ -50 dBFS); 0 = disabled
    float minDurMs  = 100.0f;   // min note duration in ms; filters brief harmonic flickers
    float ampFloor  = 0.65f;    // min note amplitude 0.0-1.0; filters weak harmonic artifacts
    float windowMs  = 300.0f;   // inference window in ms (50-2000)

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--bundle") == 0 && i + 1 < argc)
            bundlePath = argv[++i];
        else if (std::strcmp(argv[i], "--threshold") == 0 && i + 1 < argc)
            threshold = std::stof(argv[++i]);
        else if (std::strcmp(argv[i], "--gate") == 0 && i + 1 < argc)
            gateFloor = std::stof(argv[++i]);
        else if (std::strcmp(argv[i], "--min-dur") == 0 && i + 1 < argc)
            minDurMs = std::stof(argv[++i]);
        else if (std::strcmp(argv[i], "--amp-floor") == 0 && i + 1 < argc)
            ampFloor = std::stof(argv[++i]);
        else if (std::strcmp(argv[i], "--window") == 0 && i + 1 < argc)
            windowMs = std::stof(argv[++i]);
        else {
            std::fprintf(stderr,
                "Usage: %s [--bundle PATH] [--threshold 0.1-1.0] [--gate 0.0-0.1] [--min-dur MS] [--amp-floor 0.0-1.0] [--window MS]\n",
                argv[0]);
            return 1;
        }
    }

    // --- locate bundle ---
    if (bundlePath.empty()) {
        // Try next to the binary first
        std::string selfDir;
        if (argc > 0) {
            std::string self(argv[0]);
            auto slash = self.rfind('/');
            if (slash != std::string::npos)
                selfDir = self.substr(0, slash + 1);
        }
        // Probe candidate bundle directories in order of preference
        const std::string probes[][2] = {
            { selfDir,
              selfDir + "ModelData/cnn_contour_model.json" },
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
            std::fprintf(stderr,
                "Could not find model files. Use --bundle <path to lv2 bundle dir>\n");
            return 1;
        }
    }

    std::printf("Bundle:    %s\n", bundlePath.c_str());
    std::printf("Threshold: %.2f\n", threshold);
    std::printf("Gate:      %.4f (%.1f dBFS)%s\n", gateFloor,
                gateFloor > 0.0f ? 20.0f * std::log10(gateFloor) : -999.0f,
                gateFloor == 0.0f ? "  [disabled]" : "");
    std::printf("MinDur:    %.0f ms\n", minDurMs);
    std::printf("AmpFloor:  %.2f\n", ampFloor);
    std::printf("Window:    %.0f ms\n", windowMs);

    // --- load models ---
    try {
        BinaryData::init(bundlePath);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "Failed to load models: %s\n", e.what());
        return 1;
    }

    // --- build monitor ---
    Monitor mon;
    mon.threshold = threshold;
    mon.gateFloor = gateFloor;
    mon.minDurMs  = minDurMs;
    mon.ampFloor  = ampFloor;
    mon.ringSize  = windowMsToRingSize(windowMs);
    mon.bp        = std::make_unique<BasicPitch>();
    mon.startTime = std::chrono::steady_clock::now();
    g_mon         = &mon;

    // --- open JACK client ---
    jack_status_t status;
    jack_client_t* client = jack_client_open("neuralnote_monitor",
                                              JackNullOption, &status);
    if (!client) {
        std::fprintf(stderr, "Failed to connect to JACK (status 0x%x)\n", status);
        return 1;
    }

    mon.sampleRate = jack_get_sample_rate(client);
    std::printf("JACK:      %.0f Hz, buffer %u frames\n",
                mon.sampleRate, jack_get_buffer_size(client));

    jack_port_t* inPort = jack_port_register(client, "audio_in",
                                              JACK_DEFAULT_AUDIO_TYPE,
                                              JackPortIsInput, 0);
    if (!inPort) {
        std::fprintf(stderr, "Failed to register JACK input port\n");
        jack_client_close(client);
        return 1;
    }

    JackCtx ctx{ inPort, &mon };
    jack_set_process_callback(client, jackProcessImpl, &ctx);

    // --- start worker thread ---
    mon.workerThread = std::thread(runWorker, &mon);

    // --- activate JACK client ---
    if (jack_activate(client) != 0) {
        std::fprintf(stderr, "Cannot activate JACK client\n");
        {
            std::lock_guard<std::mutex> lk(mon.workerMutex);
            mon.workerQuit = true;
            mon.workerCv.notify_one();
        }
        mon.workerThread.join();
        jack_client_close(client);
        return 1;
    }

    // --- auto-connect to system capture_1 ---
    const char** ports = jack_get_ports(client, nullptr,
                                         JACK_DEFAULT_AUDIO_TYPE,
                                         JackPortIsPhysical | JackPortIsOutput);
    if (ports && ports[0]) {
        jack_connect(client, ports[0], jack_port_name(inPort));
        std::printf("Connected: %s -> neuralnote_monitor:audio_in\n\n", ports[0]);
    } else {
        std::printf("No physical capture ports found — connect manually.\n\n");
    }
    if (ports) jack_free(ports);

    std::printf("Listening for notes. Press Ctrl+C to stop.\n");
    std::printf("%-12s  %-10s  %-6s  %5s  %s\n", "Time", "Event", "Note", "MIDI#", "Vel");
    std::printf("%-12s  %-10s  %-6s  %5s  %s\n",
                "------------", "----------", "------", "-----", "---");

    // --- wait for Ctrl+C ---
    std::signal(SIGINT,  onSignal);
    std::signal(SIGTERM, onSignal);
    while (!g_quit.load())
        std::this_thread::sleep_for(std::chrono::milliseconds(50));

    // --- shut down ---
    std::printf("\nShutting down...\n");
    jack_deactivate(client);
    jack_client_close(client);

    {
        std::lock_guard<std::mutex> lk(mon.workerMutex);
        mon.workerQuit = true;
        mon.workerCv.notify_one();
    }
    mon.workerThread.join();

    return 0;
}
