/**
 * pipitch_test — Record guitar audio & regression-test PiPitch detection
 *
 * Two modes:
 *
 *   record   Capture live guitar audio from JACK and write to a WAV file
 *            on Ctrl+C.
 *
 *   test     Read a WAV file, feed it block-by-block through the full
 *            PiPitch pipeline (onset → OBP → MPM → CNN/SwiftF0), collect
 *            MIDI events, and compare against a label file.
 *
 * Usage:
 *   pipitch_test record -o guitar_e4.wav
 *   pipitch_test test   -i guitar_e4.wav -l guitar_e4.txt [--mode swiftmono] [--config ...]
 *
 * Label file format (one note per line):
 *   <onset_seconds>  <midi_note>  [duration_seconds]
 *   # comments and blank lines are ignored
 *   0.150  64  0.200
 *   0.500  62  0.300
 *
 * If duration is omitted, only onset detection is scored (no note-off check).
 */

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <csignal>
#include <memory>
#include <mutex>
#include <semaphore.h>
#include <string>
#include <thread>
#include <vector>

#include <jack/jack.h>

// MPM always enabled (Pi 5 target).
#define PIPITCH_ENABLE_MPM 1

#include "BasicPitchConstants.h"
#include "PiPitchShared.h"
#include "SwiftF0Detector.h"

// ── Note helpers ──────────────────────────────────────────────────────────────

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

// ── WAV I/O (32-bit float, mono) ─────────────────────────────────────────────

static bool writeWav(const char* path, const float* data, size_t nSamples, int sampleRate)
{
    FILE* f = std::fopen(path, "wb");
    if (!f) return false;

    const uint32_t dataSize   = static_cast<uint32_t>(nSamples * sizeof(float));
    const uint32_t fileSize   = 36 + dataSize;
    const uint16_t audioFmt   = 3;  // IEEE float
    const uint16_t numCh      = 1;
    const uint32_t sr         = static_cast<uint32_t>(sampleRate);
    const uint32_t byteRate   = sr * sizeof(float);
    const uint16_t blockAlign = sizeof(float);
    const uint16_t bitsPerSmp = 32;

    std::fwrite("RIFF", 1, 4, f);
    std::fwrite(&fileSize, 4, 1, f);
    std::fwrite("WAVE", 1, 4, f);
    std::fwrite("fmt ", 1, 4, f);
    const uint32_t fmtSize = 16;
    std::fwrite(&fmtSize, 4, 1, f);
    std::fwrite(&audioFmt, 2, 1, f);
    std::fwrite(&numCh, 2, 1, f);
    std::fwrite(&sr, 4, 1, f);
    std::fwrite(&byteRate, 4, 1, f);
    std::fwrite(&blockAlign, 2, 1, f);
    std::fwrite(&bitsPerSmp, 2, 1, f);
    std::fwrite("data", 1, 4, f);
    std::fwrite(&dataSize, 4, 1, f);
    std::fwrite(data, sizeof(float), nSamples, f);
    std::fclose(f);
    return true;
}

static bool readWav(const char* path, std::vector<float>& data, int& sampleRate)
{
    FILE* f = std::fopen(path, "rb");
    if (!f) return false;

    char riff[4]; std::fread(riff, 1, 4, f);
    if (std::memcmp(riff, "RIFF", 4) != 0) { std::fclose(f); return false; }

    uint32_t fileSize; std::fread(&fileSize, 4, 1, f);
    char wave[4]; std::fread(wave, 1, 4, f);
    if (std::memcmp(wave, "WAVE", 4) != 0) { std::fclose(f); return false; }

    uint16_t audioFmt = 0, numCh = 0, bitsPerSmp = 0;
    uint32_t sr = 0;

    // Read chunks until we find "data"
    while (!std::feof(f)) {
        char chunkId[4]; if (std::fread(chunkId, 1, 4, f) != 4) break;
        uint32_t chunkSize; if (std::fread(&chunkSize, 4, 1, f) != 1) break;

        if (std::memcmp(chunkId, "fmt ", 4) == 0) {
            long fmtStart = std::ftell(f);
            std::fread(&audioFmt, 2, 1, f);
            std::fread(&numCh, 2, 1, f);
            std::fread(&sr, 4, 1, f);
            uint32_t byteRate; std::fread(&byteRate, 4, 1, f);
            uint16_t blockAlign; std::fread(&blockAlign, 2, 1, f);
            std::fread(&bitsPerSmp, 2, 1, f);
            std::fseek(f, fmtStart + chunkSize, SEEK_SET);
        } else if (std::memcmp(chunkId, "data", 4) == 0) {
            sampleRate = static_cast<int>(sr);

            if (audioFmt == 3 && bitsPerSmp == 32) {
                // IEEE float
                size_t nSamples = chunkSize / sizeof(float);
                if (numCh > 1) {
                    // Read interleaved, take first channel
                    size_t nFrames = nSamples / numCh;
                    std::vector<float> interleaved(nSamples);
                    std::fread(interleaved.data(), sizeof(float), nSamples, f);
                    data.resize(nFrames);
                    for (size_t i = 0; i < nFrames; ++i)
                        data[i] = interleaved[i * numCh];
                } else {
                    data.resize(nSamples);
                    std::fread(data.data(), sizeof(float), nSamples, f);
                }
            } else if (audioFmt == 1 && bitsPerSmp == 16) {
                // PCM 16-bit
                size_t nSamplesTotal = chunkSize / 2;
                size_t nFrames = (numCh > 1) ? nSamplesTotal / numCh : nSamplesTotal;
                std::vector<int16_t> raw(nSamplesTotal);
                std::fread(raw.data(), 2, nSamplesTotal, f);
                data.resize(nFrames);
                for (size_t i = 0; i < nFrames; ++i)
                    data[i] = raw[i * numCh] / 32768.0f;
            } else if (audioFmt == 1 && bitsPerSmp == 24) {
                // PCM 24-bit
                size_t nSamplesTotal = chunkSize / 3;
                size_t nFrames = (numCh > 1) ? nSamplesTotal / numCh : nSamplesTotal;
                std::vector<uint8_t> raw(chunkSize);
                std::fread(raw.data(), 1, chunkSize, f);
                data.resize(nFrames);
                for (size_t i = 0; i < nFrames; ++i) {
                    size_t idx = i * numCh * 3;
                    int32_t val = (raw[idx] | (raw[idx+1] << 8) | (raw[idx+2] << 16));
                    if (val & 0x800000) val |= 0xFF000000;  // sign extend
                    data[i] = val / 8388608.0f;
                }
            } else {
                std::fprintf(stderr, "Unsupported WAV format: fmt=%d bits=%d\n", audioFmt, bitsPerSmp);
                std::fclose(f); return false;
            }
            std::fclose(f);
            return true;
        } else {
            std::fseek(f, chunkSize, SEEK_CUR);
        }
    }
    std::fclose(f);
    return false;
}

// ── Note name parsing ─────────────────────────────────────────────────────────

static int parseNoteName(const char* s)
{
    // Parse "C#4", "E2", "Bb3", "64" etc.  Returns MIDI number or -1.
    // Try plain integer first.
    char* end = nullptr;
    long val = std::strtol(s, &end, 10);
    if (end != s && (*end == '\0' || *end == '\n' || *end == '\r' || *end == ' ')
        && val >= 0 && val <= 127)
        return static_cast<int>(val);

    static const char* names[] = {"C","C#","Db","D","D#","Eb","E","F","F#","Gb","G","G#","Ab","A","A#","Bb","B"};
    static const int   semi[]  = { 0,  1,   1,  2,  3,   3,  4, 5,  6,   6,  7,  8,   8,  9, 10,  10, 11};
    int noteIdx = -1, nameLen = 0;
    for (int i = 0; i < 17; ++i) {
        int len = static_cast<int>(std::strlen(names[i]));
        if (std::strncmp(s, names[i], len) == 0 && len > nameLen)
            { noteIdx = i; nameLen = len; }
    }
    if (noteIdx >= 0 && s[nameLen] >= '0' && s[nameLen] <= '9')
        return (s[nameLen] - '0' + 1) * 12 + semi[noteIdx];
    return -1;
}

// ── Chord expansion ──────────────────────────────────────────────────────────
// Expands "Am", "G", "Dmaj", "E7" etc. into constituent MIDI notes.
// Root is placed in octave 3 by default; intervals define the chord quality.

static std::vector<int> expandChord(const char* name)
{
    // Parse root note (no octave number)
    static const char* roots[]  = {"C","C#","Db","D","D#","Eb","E","F","F#","Gb","G","G#","Ab","A","A#","Bb","B"};
    static const int   rSemi[]  = { 0,  1,   1,  2,  3,   3,  4, 5,  6,   6,  7,  8,   8,  9, 10,  10, 11};
    int rootSemi = -1, rootLen = 0;
    for (int i = 0; i < 17; ++i) {
        int len = static_cast<int>(std::strlen(roots[i]));
        if (std::strncmp(name, roots[i], len) == 0 && len > rootLen)
            { rootSemi = rSemi[i]; rootLen = len; }
    }
    if (rootSemi < 0) return {};

    const char* qual = name + rootLen;
    // Default octave 3 for root (MIDI = (3+1)*12 + semi = 48 + semi)
    const int rootMidi = 48 + rootSemi;

    // Interval sets (semitones above root)
    std::vector<int> intervals;
    if (!std::strcmp(qual, "m") || !std::strcmp(qual, "min"))
        intervals = {0, 3, 7};                      // minor triad
    else if (!std::strcmp(qual, "7"))
        intervals = {0, 4, 7, 10};                  // dominant 7th
    else if (!std::strcmp(qual, "m7") || !std::strcmp(qual, "min7"))
        intervals = {0, 3, 7, 10};                  // minor 7th
    else if (!std::strcmp(qual, "maj7") || !std::strcmp(qual, "M7"))
        intervals = {0, 4, 7, 11};                  // major 7th
    else if (!std::strcmp(qual, "dim"))
        intervals = {0, 3, 6};                      // diminished triad
    else if (!std::strcmp(qual, "aug") || !std::strcmp(qual, "+"))
        intervals = {0, 4, 8};                      // augmented triad
    else if (!std::strcmp(qual, "sus2"))
        intervals = {0, 2, 7};                      // suspended 2nd
    else if (!std::strcmp(qual, "sus4"))
        intervals = {0, 5, 7};                      // suspended 4th
    else if (!std::strcmp(qual, "5") || !std::strcmp(qual, "power"))
        intervals = {0, 7};                          // power chord
    else
        intervals = {0, 4, 7};                      // major triad (default)

    std::vector<int> notes;
    for (int iv : intervals)
        notes.push_back(rootMidi + iv);
    return notes;
}

// ── Label file ────────────────────────────────────────────────────────────────
//
// Format:
//   First non-comment line: "mono" or "chord"
//   Subsequent lines: one entry per line
//
// mono mode — each line is a note name or MIDI number:
//   E4
//   D4
//   E4
//
// chord mode — each line is a chord symbol:
//   Am
//   G
//   Dmaj
//
// The order of lines is the expected order of notes/chords in the recording.
// Comments (#) and blank lines are ignored.

enum class LabelMode { MONO, CHORD };

struct LabelEntry {
    std::string        text;      // original line text
    std::vector<int>   midiNotes; // expected MIDI notes (1 for mono, N for chord)
};

struct LabelFile {
    LabelMode                mode = LabelMode::MONO;
    std::vector<LabelEntry>  entries;
};

static LabelFile loadLabels(const char* path)
{
    LabelFile lf;
    FILE* f = std::fopen(path, "r");
    if (!f) return lf;

    bool modeRead = false;
    char line[256];
    while (std::fgets(line, sizeof(line), f)) {
        // Trim leading whitespace
        char* p = line;
        while (*p == ' ' || *p == '\t') ++p;
        // Strip trailing newline/whitespace
        char* end = p + std::strlen(p);
        while (end > p && (end[-1] == '\n' || end[-1] == '\r' || end[-1] == ' ')) --end;
        *end = '\0';
        if (*p == '#' || *p == '\0') continue;

        if (!modeRead) {
            if (!std::strcmp(p, "mono"))       lf.mode = LabelMode::MONO;
            else if (!std::strcmp(p, "chord")) lf.mode = LabelMode::CHORD;
            else { std::fprintf(stderr, "Label error: first line must be 'mono' or 'chord', got '%s'\n", p); break; }
            modeRead = true;
            continue;
        }

        LabelEntry entry;
        entry.text = p;

        if (lf.mode == LabelMode::MONO) {
            int midi = parseNoteName(p);
            if (midi >= 0)
                entry.midiNotes.push_back(midi);
            else
                std::fprintf(stderr, "Warning: cannot parse note '%s'\n", p);
        } else {
            entry.midiNotes = expandChord(p);
            if (entry.midiNotes.empty())
                std::fprintf(stderr, "Warning: cannot parse chord '%s'\n", p);
        }

        if (!entry.midiNotes.empty())
            lf.entries.push_back(std::move(entry));
    }
    std::fclose(f);
    return lf;
}

// ── MIDI event log (collected during test) ────────────────────────────────────

struct MidiEvent {
    double timeS;
    int    midiNote;
    bool   noteOn;
    int    velocity;
};

// ══════════════════════════════════════════════════════════════════════════════
//  RECORD MODE — JACK capture → WAV
// ══════════════════════════════════════════════════════════════════════════════

static std::atomic<bool> g_quit{false};
static void onSignal(int) { g_quit.store(true); }

struct RecordCtx {
    jack_port_t*       inPort = nullptr;
    std::mutex         mtx;
    std::vector<float> samples;
};

static int recordCallback(jack_nframes_t nFrames, void* arg)
{
    auto* ctx = static_cast<RecordCtx*>(arg);
    const float* in = static_cast<const float*>(
        jack_port_get_buffer(ctx->inPort, nFrames));

    std::lock_guard<std::mutex> lock(ctx->mtx);
    ctx->samples.insert(ctx->samples.end(), in, in + nFrames);
    return 0;
}

static int runRecord(const char* outPath, const char* jackPortName)
{
    jack_status_t status;
    jack_client_t* client = jack_client_open("pipitch_record", JackNullOption, &status);
    if (!client) {
        std::fprintf(stderr, "Cannot open JACK client (0x%x)\n", status);
        return 1;
    }

    const int sr = static_cast<int>(jack_get_sample_rate(client));
    std::printf("JACK:    %d Hz, buffer %u frames\n", sr, jack_get_buffer_size(client));

    RecordCtx ctx;
    ctx.inPort = jack_port_register(client, "audio_in",
                                     JACK_DEFAULT_AUDIO_TYPE, JackPortIsInput, 0);
    if (!ctx.inPort) {
        std::fprintf(stderr, "Cannot register JACK port\n");
        jack_client_close(client); return 1;
    }

    jack_set_process_callback(client, recordCallback, &ctx);

    if (jack_activate(client) != 0) {
        std::fprintf(stderr, "Cannot activate JACK client\n");
        jack_client_close(client); return 1;
    }

    // Connect to specified port or first physical capture port
    if (jackPortName && jackPortName[0]) {
        if (jack_connect(client, jackPortName, jack_port_name(ctx.inPort)) != 0)
            std::fprintf(stderr, "Warning: cannot connect to %s\n", jackPortName);
        else
            std::printf("Input:   %s\n", jackPortName);
    } else {
        const char** caps = jack_get_ports(client, nullptr, JACK_DEFAULT_AUDIO_TYPE,
                                            JackPortIsPhysical | JackPortIsOutput);
        if (caps && caps[0]) {
            jack_connect(client, caps[0], jack_port_name(ctx.inPort));
            std::printf("Input:   %s\n", caps[0]);
        }
        if (caps) jack_free(caps);
    }

    std::printf("Output:  %s\n", outPath);
    std::printf("\nRecording. Press Ctrl+C to stop and save.\n");

    std::signal(SIGINT,  onSignal);
    std::signal(SIGTERM, onSignal);
    while (!g_quit.load())
        std::this_thread::sleep_for(std::chrono::milliseconds(50));

    jack_deactivate(client);
    jack_client_close(client);

    std::lock_guard<std::mutex> lock(ctx.mtx);
    const size_t n = ctx.samples.size();
    const double durS = static_cast<double>(n) / sr;
    std::printf("\nCaptured %.2f seconds (%zu samples)\n", durS, n);

    if (n == 0) {
        std::printf("No audio captured — skipping WAV write.\n");
        return 0;
    }

    if (writeWav(outPath, ctx.samples.data(), n, sr)) {
        std::printf("Written to %s\n", outPath);
        return 0;
    } else {
        std::fprintf(stderr, "Failed to write %s\n", outPath);
        return 1;
    }
}

// ══════════════════════════════════════════════════════════════════════════════
//  TEST MODE — WAV → PiPitch pipeline → compare with labels
// ══════════════════════════════════════════════════════════════════════════════

// Resampler (same as pipitch_tune.cpp)
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

// Per-range state for test mode (minimal — no synth)
struct TestRangeState : RangeStateBase {
    int  lastSwiftPrint = -2;
};

// Shared state for test mode
struct TestState {
    double   sampleRate     = 48000.0;
    float    gateFloor      = 0.003f;
    float    ampFloor       = 0.65f;
    float    threshold      = 0.6f;
    float    frameThreshold = 0.5f;
    float    onsetBlankMs   = 25.0f;
    float    swiftF0Threshold = 0.5f;
    float    octaveLockMs   = 250.0f;
    bool     bendEnabled    = false;
    PlayMode mode           = PlayMode::MONO;
    ProvMode provisionalMode = ProvMode::ON;

    std::vector<std::unique_ptr<TestRangeState>> ranges;

    std::chrono::steady_clock::time_point startTime;

    // Audio-domain time tracking (for accurate scoring vs labels)
    std::atomic<double> audioTimeS{0.0};  // updated after each processTestBlock

    // Onset detector state
    float onsetSmoothedRms = 0.001f;
    int   onsetBlankRemain = 0;
    PickDetector pickDetector;
    int   pickFiredRemain  = 0;
    float lastPickRatio    = 0.0f;
    std::atomic<uint64_t> totalSamples{0};
    std::atomic<uint64_t> lastOnsetSample{0};

    std::unique_ptr<SwiftF0Detector> swiftF0;
#ifdef __aarch64__
    UltraLowLatencyGoertzel goertzel;
    uint64_t goertzelPrevBits = 0;
    int      goertzelNoteWindow = 0;
    int      goertzelPolyOnCount = 0;
#endif
    int      maxPoly = 3;
    std::vector<float> sf0Buf;

    std::thread       workerThread;
    std::atomic<bool> workerQuit{false};
    sem_t             workerSem;

    // Collected MIDI events (written from both audio sim + worker via midiOut)
    std::mutex             eventsMtx;
    std::vector<MidiEvent> events;
};

// Worker hooks for test mode — minimal logging, collect events
struct TestWorkerHooks {
    TestState* st;

    sem_t&              workerSem()      { return st->workerSem; }
    bool                shouldQuit()     { return st->workerQuit.load(std::memory_order_acquire); }
    float               ampFloor()       { return st->ampFloor; }
    int                 mode()           { return static_cast<int>(st->mode); }
    float               frameThreshold() { return st->frameThreshold; }
    float               threshold()      { return st->threshold; }
    float               swiftThreshold() { return st->swiftF0Threshold; }
    double              sampleRate()     { return st->sampleRate; }
    SwiftF0Detector*    swiftF0()        { return st->swiftF0.get(); }
    std::vector<float>& sf0Buf()         { return st->sf0Buf; }
    uint64_t            totalSamples()   { return st->totalSamples.load(std::memory_order_relaxed); }
    uint64_t            lastOnsetSample(){ return st->lastOnsetSample.load(std::memory_order_acquire); }
    int                 provisionalMode(){ return static_cast<int>(st->provisionalMode); }
    float               octaveLockMs()   { return st->octaveLockMs; }
    bool                bendEnabled()    { return st->bendEnabled; }
    auto&               ranges()         { return st->ranges; }

    double elapsed() const {
        // Use audio-domain time for accurate scoring against labels.
        return st->audioTimeS.load(std::memory_order_relaxed);
    }

    void onGoertzelPolyResult(TestRangeState&, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, double) {}
    void onSwiftResult(TestRangeState& r, int effectiveNote, double inferMs) {
        if (effectiveNote != r.lastSwiftPrint) {
            r.lastSwiftPrint = effectiveNote;
            if (effectiveNote >= 0)
                std::printf("  SwiftF0 → %-4s (%3d)  [inf %4.0fms  range %s]\n",
                            midiToName(effectiveNote).c_str(), effectiveNote,
                            inferMs, r.cfg.name.c_str());
        }
    }

    void onSwiftPolyResult(TestRangeState& r, int swiftNote, double sf0Ms,
                           uint64_t cnnBits, double cnnMs) {
        (void)r; (void)swiftNote; (void)sf0Ms; (void)cnnBits; (void)cnnMs;
    }

    void onCNNOutcome(TestRangeState& r, int prov, uint64_t newBits, double inferMs) {
        (void)r; (void)prov; (void)newBits; (void)inferMs;
    }

    void onNotesChanged(TestRangeState& r, uint64_t prevActive,
                        const int8_t* newVel, double inferMs, const char* modeLabel) {
        const double el = elapsed();
        (void)inferMs; (void)modeLabel;
        for (uint64_t tmp = r.activeNotes & ~prevActive; tmp; tmp &= tmp - 1) {
            const int bit = __builtin_ctzll(tmp);
            const int p   = NOTE_BASE + bit;
            std::lock_guard<std::mutex> lock(st->eventsMtx);
            st->events.push_back({el, p, true, static_cast<int>(newVel[bit])});
        }
        for (uint64_t tmp = prevActive & ~r.activeNotes; tmp; tmp &= tmp - 1) {
            const int p = NOTE_BASE + static_cast<int>(__builtin_ctzll(tmp));
            std::lock_guard<std::mutex> lock(st->eventsMtx);
            st->events.push_back({el, p, false, 0});
        }
    }

    void onMonoKill(TestRangeState& r, int pitch) {
        (void)r;
        std::lock_guard<std::mutex> lock(st->eventsMtx);
        st->events.push_back({elapsed(), pitch, false, 0});
    }

    void onShutdownOff(TestRangeState& r, int pitch) {
        (void)r;
        std::lock_guard<std::mutex> lock(st->eventsMtx);
        st->events.push_back({elapsed(), pitch, false, 0});
    }
};

static void testWorker(TestState* st)
{
    TestWorkerHooks hooks{st};
    runWorkerCommon(hooks);
}

// Simulate the audio callback processing for one block
static void processTestBlock(TestState* st, const float* blockIn, int nSamples)
{
    // RMS + gate
    float sumSq = 0.0f;
    for (int i = 0; i < nSamples; ++i)
        sumSq += blockIn[i] * blockIn[i];
    const float blockRms = std::sqrt(sumSq / static_cast<float>(nSamples));
    const float gateFloor = st->gateFloor;
    const bool  gated = (blockRms < gateFloor);

    // Onset detection
    st->totalSamples.fetch_add(nSamples, std::memory_order_relaxed);
    if (st->pickFiredRemain > 0)
        st->pickFiredRemain -= nSamples;
    bool onsetFired = false;

    constexpr float PICK_HIGH_TIER = 10.0f;
    int   pickSample = -1;
    float pickRatio  = 0.0f;
    if (!gated)
        pickSample = st->pickDetector.process(blockIn, nSamples, pickRatio);

    const bool rmsWouldFire = !gated && st->onsetBlankRemain <= 0
                              && blockRms > st->onsetSmoothedRms * ONSET_RATIO;

    if (pickSample >= 0 && (pickRatio >= PICK_HIGH_TIER || rmsWouldFire)) {
        onsetFired = true;
        for (auto& rp : st->ranges)
            { rp->provCooldownRemain = 0; rp->provCooldownNote = -1; }
        st->pickFiredRemain = static_cast<int>(st->sampleRate * 0.05);
        st->lastPickRatio   = pickRatio;
        st->onsetBlankRemain = static_cast<int>(st->sampleRate * (st->onsetBlankMs / 1000.0f));
        st->onsetSmoothedRms = blockRms;
        st->lastOnsetSample.store(
            st->totalSamples.load(std::memory_order_relaxed)
                - static_cast<uint64_t>(nSamples) + static_cast<uint64_t>(pickSample),
            std::memory_order_release);
    }

    if (!onsetFired) {
        if (st->onsetBlankRemain > 0) {
            st->onsetBlankRemain -= nSamples;
            if (st->onsetBlankRemain < 0) st->onsetBlankRemain = 0;
        } else if (rmsWouldFire) {
            onsetFired = true;
            st->onsetBlankRemain = static_cast<int>(st->sampleRate * (st->onsetBlankMs / 1000.0f));
            st->onsetSmoothedRms = blockRms;
            st->lastOnsetSample.store(st->totalSamples.load(std::memory_order_relaxed),
                                      std::memory_order_release);
        }
    }
    if (!onsetFired && st->onsetBlankRemain == 0)
        st->onsetSmoothedRms = st->onsetSmoothedRms * (1.0f - ONSET_ALPHA) + blockRms * ONSET_ALPHA;

    // Resample to 22050 Hz
    std::vector<float> resampled;
    resampled.reserve(static_cast<int>(nSamples * (PLUGIN_SR / st->sampleRate)) + 2);
    if (gated) resampled.assign(static_cast<int>(nSamples * (PLUGIN_SR / st->sampleRate)), 0.0f);
    else       resampleLinear(blockIn, nSamples, st->sampleRate, resampled);

    // GoertzelMono bypasses ring/OBP/CNN pipeline
    const bool goertzelMono = (st->mode == PlayMode::GOERTZEL_MONO);
    const bool goertzelPoly = (st->mode == PlayMode::GOERTZEL_POLY);

    for (auto& rp : st->ranges) {
        TestRangeState& r = *rp;
        if (goertzelMono) continue;

        // GoertzelPoly: feed ring and dispatch for CNN, skip OBP/provisional
        if (goertzelPoly) {
            pushRingSamples(r, resampled.data(), static_cast<int>(resampled.size()));
            dispatchSnapshotIfReady(r, onsetFired, 0.0, st->workerSem, gateFloor);
            continue;
        }

        if (!gated) {
            if (onsetFired) {
                std::memset(r.ring.data(), 0, r.ringSize * sizeof(float));
                r.freshSamples = 0;
            }
            pushRingSamples(r, resampled.data(), static_cast<int>(resampled.size()));

            const bool provEnabled = (st->provisionalMode == ProvMode::ON
                                      || st->provisionalMode == ProvMode::ADAPTIVE);
            if (provEnabled) {
                if (onsetFired && st->pickFiredRemain > 0)
                    r.obpBlankRemain = static_cast<int>(st->sampleRate * 0.005);
                const bool obpBlanked = (r.obpBlankRemain > 0);
                if (obpBlanked) r.obpBlankRemain -= nSamples;

                armOrExpireOBP(r, static_cast<float>(st->sampleRate), nSamples, onsetFired);

                if (r.provCooldownRemain > 0)
                    r.provCooldownRemain -= nSamples;

                if (!gated && (r.obdOnsetActive || r.obdPendingNote != -1))
                    r.mpm.push(blockIn, nSamples);

                if (r.obdOnsetActive && !obpBlanked) {
                    if (r.provNote.load(std::memory_order_relaxed) == -1) {
                        const int finalNote = runOBPHPS(r, blockIn, nSamples,
                                                        static_cast<float>(st->sampleRate),
                                                        st->ranges);
                        if (finalNote == -1 && !r.obdOnsetActive) {
                            // OBP expired — try MPM fallback
                            const int mpmNote = r.mpm.analyze(static_cast<float>(st->sampleRate),
                                                               r.cfg.midiLow, r.cfg.midiHigh);
                            if (mpmNote != -1) {
                                r.provNote.store(mpmNote, std::memory_order_release);
                                r.monoHeldNote.store(mpmNote, std::memory_order_release);
                            }
                        }
                        if (finalNote != -1) {
                            const int mpmNote = r.mpm.analyze(static_cast<float>(st->sampleRate),
                                                               r.cfg.midiLow, r.cfg.midiHigh);
                            if (mpmNote == -1) {
                                r.obdPendingNote   = finalNote;
                                r.obdPendingRemain = static_cast<int>(st->sampleRate * 0.1f);
                            } else {
                                const int agreed = mpmNote;  // trust MPM
                                r.provNote.store(agreed, std::memory_order_release);
                                r.monoHeldNote.store(agreed, std::memory_order_release);
                            }
                        }
                    }
                    // Pending MPM retry
                    if (r.obdPendingNote != -1) {
                        r.obdPendingRemain -= nSamples;
                        const int mpmNote = r.mpm.analyze(static_cast<float>(st->sampleRate),
                                                           r.cfg.midiLow, r.cfg.midiHigh);
                        if (mpmNote != -1) {
                            r.provNote.store(mpmNote, std::memory_order_release);
                            r.monoHeldNote.store(mpmNote, std::memory_order_release);
                            r.obdPendingNote = -1;
                        } else if (r.obdPendingRemain <= 0) {
                            r.obdPendingNote = -1;
                        }
                    }
                }
            } // provEnabled
        } else {
            resetOBPOnGate(r);
            // Push resampled zeros (already at 22050 Hz)
            pushRingSamples(r, resampled.data(), static_cast<int>(resampled.size()));
        }

        dispatchSnapshotIfReady(r, onsetFired, 0.0, st->workerSem, gateFloor);
    }

    // GoertzelMono audio-thread processing
#ifdef __aarch64__
    if (st->mode == PlayMode::GOERTZEL_MONO && !gated) {
        st->goertzel.processBlock(blockIn, nSamples, onsetFired);

        auto& states = st->goertzel.getNoteStates();
        const int gStart = st->goertzel.startMidi();
        const double el = st->audioTimeS.load(std::memory_order_relaxed);

        for (int i = 0; i < st->goertzel.numNotes(); ++i) {
            const int midi = gStart + i;
            if (midi < NOTE_BASE || midi >= NOTE_BASE + NOTE_COUNT) continue;
            auto& s = states[i];
            const uint64_t bit = 1ULL << (midi - NOTE_BASE);

            if (s.isActive() && !(st->goertzelPrevBits & bit)) {
                if (s.triggerPending) {
                    bool octLocked = false;
                    for (uint64_t tmp = st->goertzelPrevBits; tmp; tmp &= tmp - 1) {
                        const int act = NOTE_BASE + __builtin_ctzll(tmp);
                        const int diff = std::abs(midi - act);
                        if (diff == 12 || diff == 24) { octLocked = true; break; }
                    }
                    if (!octLocked) {
                        std::lock_guard<std::mutex> lock(st->eventsMtx);
                        st->events.push_back({el, midi, true, s.velocity});
                        st->goertzelPrevBits |= bit;
                    }
                    s.triggerPending = false;
                }
            } else if (!s.isActive() && (st->goertzelPrevBits & bit)) {
                std::lock_guard<std::mutex> lock(st->eventsMtx);
                st->events.push_back({el, midi, false, 0});
                st->goertzelPrevBits &= ~bit;
            }
        }
    } else if (st->mode == PlayMode::GOERTZEL_MONO && gated) {
        st->goertzel.drainGated(nSamples);
        auto& states = st->goertzel.getNoteStates();
        const int gStart = st->goertzel.startMidi();
        const double el = st->audioTimeS.load(std::memory_order_relaxed);
        for (int i = 0; i < st->goertzel.numNotes(); ++i) {
            const int midi = gStart + i;
            if (midi < NOTE_BASE || midi >= NOTE_BASE + NOTE_COUNT) continue;
            const uint64_t bit = 1ULL << (midi - NOTE_BASE);
            if (!states[i].isActive() && (st->goertzelPrevBits & bit)) {
                std::lock_guard<std::mutex> lock(st->eventsMtx);
                st->events.push_back({el, midi, false, 0});
                st->goertzelPrevBits &= ~bit;
            }
        }
    }

    // GoertzelPoly audio-thread processing
    if (goertzelPoly && !gated) {
        st->goertzel.processBlock(blockIn, nSamples, onsetFired, true);

        auto& states = st->goertzel.getNoteStates();
        const int gStart = st->goertzel.startMidi();
        const double el = st->audioTimeS.load(std::memory_order_relaxed);

        if (onsetFired) {
            if (st->goertzelNoteWindow <= 0)
                st->goertzelPolyOnCount = 0;
            st->goertzelNoteWindow = static_cast<int>(st->sampleRate * 0.05);  // 50ms
        }
        if (st->goertzelNoteWindow > 0)
            st->goertzelNoteWindow -= nSamples;

        for (int i = 0; i < st->goertzel.numNotes(); ++i) {
            const int midi = gStart + i;
            if (midi < NOTE_BASE || midi >= NOTE_BASE + NOTE_COUNT) continue;
            auto& s = states[i];
            const uint64_t bit = 1ULL << (midi - NOTE_BASE);

            if (s.isActive() && s.triggerPending && !(st->goertzelPrevBits & bit)) {
                s.triggerPending = false;
                if (st->goertzelNoteWindow > 0 && st->goertzelPolyOnCount < st->maxPoly) {
                    std::lock_guard<std::mutex> lock(st->eventsMtx);
                    st->events.push_back({el, midi, true, 40});
                    st->goertzelPrevBits |= bit;
                    st->goertzelPolyOnCount++;
                }
            }
            if (!s.isActive() && (st->goertzelPrevBits & bit)) {
                std::lock_guard<std::mutex> lock(st->eventsMtx);
                st->events.push_back({el, midi, false, 0});
                st->goertzelPrevBits &= ~bit;
            }
        }

        for (auto& rp : st->ranges)
            rp->goertzelPolyActiveBits.store(st->goertzelPrevBits, std::memory_order_release);

    } else if (goertzelPoly && gated) {
        st->goertzel.drainGated(nSamples);
        auto& states = st->goertzel.getNoteStates();
        const int gStart = st->goertzel.startMidi();
        const double el = st->audioTimeS.load(std::memory_order_relaxed);
        for (int i = 0; i < st->goertzel.numNotes(); ++i) {
            const int midi = gStart + i;
            if (midi < NOTE_BASE || midi >= NOTE_BASE + NOTE_COUNT) continue;
            const uint64_t bit = 1ULL << (midi - NOTE_BASE);
            if (!states[i].isActive() && (st->goertzelPrevBits & bit)) {
                std::lock_guard<std::mutex> lock(st->eventsMtx);
                st->events.push_back({el, midi, false, 0});
                st->goertzelPrevBits &= ~bit;
            }
        }
        for (auto& rp : st->ranges)
            rp->goertzelPolyActiveBits.store(st->goertzelPrevBits, std::memory_order_release);
    }
#endif

    // Drain midiOut queues from worker (notes pushed by worker)
    const double audioT = st->audioTimeS.load(std::memory_order_relaxed);
    for (auto& rp : st->ranges) {
        PendingNote pn;
        while (rp->midiOut.pop(pn)) {
            if (pn.type == PendingNote::PITCH_BEND) continue;
            std::lock_guard<std::mutex> lock(st->eventsMtx);
            const bool isOn = (pn.type == PendingNote::NOTE_ON);
            st->events.push_back({audioT, pn.pitch, isOn, pn.value});
        }
    }

    // Advance audio-domain clock
    st->audioTimeS.store(
        st->audioTimeS.load(std::memory_order_relaxed) + nSamples / st->sampleRate,
        std::memory_order_release);
}

// ── Segment events by silence gaps ────────────────────────────────────────────
// A "segment" is a group of MIDI activity between silence gaps.
// We detect segments by tracking when all notes are off for > gapThreshold.

struct Segment {
    double           startS;
    double           endS;
    std::vector<int> notesDetected;  // unique MIDI notes that had note-ON in this segment
};

static std::vector<Segment> segmentEvents(const std::vector<MidiEvent>& events,
                                           double gapThresholdS = 0.300)
{
    std::vector<Segment> segs;
    int activeCount = 0;
    double lastOffTime = 0.0;
    Segment cur{};
    bool inSeg = false;

    for (const auto& e : events) {
        if (e.noteOn) {
            if (!inSeg) {
                // Start new segment
                cur = Segment{};
                cur.startS = e.timeS;
                inSeg = true;
            } else if (activeCount == 0 && (e.timeS - lastOffTime) > gapThresholdS) {
                // Gap detected — close old segment, start new one
                cur.endS = lastOffTime;
                segs.push_back(cur);
                cur = Segment{};
                cur.startS = e.timeS;
            }
            activeCount++;
            // Add note if not already present
            if (std::find(cur.notesDetected.begin(), cur.notesDetected.end(),
                          e.midiNote) == cur.notesDetected.end())
                cur.notesDetected.push_back(e.midiNote);
        } else {
            if (activeCount > 0) activeCount--;
            lastOffTime = e.timeS;
        }
    }
    if (inSeg) {
        cur.endS = lastOffTime;
        segs.push_back(cur);
    }
    return segs;
}

// ── Score matching ────────────────────────────────────────────────────────────

struct TestResult {
    int totalEntries    = 0;
    int hits            = 0;   // all expected notes found in the segment
    int partialHits     = 0;   // some but not all expected notes found
    int misses          = 0;   // no expected notes found
    int wrongNotes      = 0;   // unexpected notes in matched segments
    int extraSegments   = 0;   // segments with no matching label
    int missingSegments = 0;   // labels with no matching segment

    std::vector<std::string> details;
};

static TestResult scoreEvents(const std::vector<MidiEvent>& events,
                               const LabelFile& lf)
{
    TestResult res;
    res.totalEntries = static_cast<int>(lf.entries.size());

    std::vector<Segment> segs = segmentEvents(events);

    const int nLabels = static_cast<int>(lf.entries.size());
    const int nSegs   = static_cast<int>(segs.size());

    // Match segments to labels in order (1:1 by position)
    const int matched = std::min(nLabels, nSegs);
    for (int i = 0; i < matched; ++i) {
        const auto& entry = lf.entries[i];
        const auto& seg   = segs[i];

        // Check which expected notes were found
        int found = 0, missing = 0;
        std::string foundStr, missStr;
        for (int expected : entry.midiNotes) {
            if (std::find(seg.notesDetected.begin(), seg.notesDetected.end(),
                          expected) != seg.notesDetected.end()) {
                found++;
                if (!foundStr.empty()) foundStr += " ";
                foundStr += midiToName(expected);
            } else {
                missing++;
                if (!missStr.empty()) missStr += " ";
                missStr += midiToName(expected);
            }
        }

        // Check for unexpected notes
        int extra = 0;
        std::string extraStr;
        for (int det : seg.notesDetected) {
            if (std::find(entry.midiNotes.begin(), entry.midiNotes.end(),
                          det) == entry.midiNotes.end()) {
                extra++;
                if (!extraStr.empty()) extraStr += " ";
                extraStr += midiToName(det);
            }
        }
        res.wrongNotes += extra;

        char buf[512];
        if (missing == 0) {
            res.hits++;
            if (extra == 0)
                std::snprintf(buf, sizeof(buf), "  HIT    [%d] %-12s  detected: %s",
                              i + 1, entry.text.c_str(), foundStr.c_str());
            else
                std::snprintf(buf, sizeof(buf), "  HIT    [%d] %-12s  detected: %s  extra: %s",
                              i + 1, entry.text.c_str(), foundStr.c_str(), extraStr.c_str());
        } else if (found > 0) {
            res.partialHits++;
            std::snprintf(buf, sizeof(buf), "  PARTIAL[%d] %-12s  found: %s  missing: %s%s",
                          i + 1, entry.text.c_str(), foundStr.c_str(), missStr.c_str(),
                          extra > 0 ? ("  extra: " + extraStr).c_str() : "");
        } else {
            res.misses++;
            std::snprintf(buf, sizeof(buf), "  MISS   [%d] %-12s  got: ", i + 1, entry.text.c_str());
            std::string b(buf);
            for (int det : seg.notesDetected)
                b += midiToName(det) + " ";
            std::strncpy(buf, b.c_str(), sizeof(buf) - 1);
        }
        res.details.push_back(buf);
    }

    // Unmatched labels (more labels than segments)
    for (int i = matched; i < nLabels; ++i) {
        res.missingSegments++;
        char buf[256];
        std::snprintf(buf, sizeof(buf), "  NOSEG  [%d] %-12s  (no audio segment detected)",
                      i + 1, lf.entries[i].text.c_str());
        res.details.push_back(buf);
    }

    // Extra segments (more segments than labels)
    for (int i = matched; i < nSegs; ++i) {
        res.extraSegments++;
        char buf[256];
        std::string notes;
        for (int n : segs[i].notesDetected) notes += midiToName(n) + " ";
        std::snprintf(buf, sizeof(buf), "  EXTRA  [%d] %.3f-%.3fs  detected: %s",
                      i + 1, segs[i].startS, segs[i].endS, notes.c_str());
        res.details.push_back(buf);
    }

    return res;
}

static int runTest(const char* inputPath, const char* labelPath,
                   const std::string& bundlePath, const std::string& configPath,
                   const RangeConfig& rangeCfg, int blockSize)
{
    // Load WAV
    std::vector<float> wavData;
    int wavSr = 0;
    if (!readWav(inputPath, wavData, wavSr)) {
        std::fprintf(stderr, "Cannot read WAV file: %s\n", inputPath);
        return 1;
    }
    std::printf("WAV:     %s  (%.2f s, %d Hz, %zu samples)\n",
                inputPath, static_cast<double>(wavData.size()) / wavSr, wavSr, wavData.size());

    // Load labels
    LabelFile lf = loadLabels(labelPath);
    if (lf.entries.empty()) {
        std::fprintf(stderr, "No labels loaded from: %s\n", labelPath);
        return 1;
    }
    std::printf("Labels:  %s  (%s, %zu entries)\n", labelPath,
                lf.mode == LabelMode::MONO ? "mono" : "chord",
                lf.entries.size());
    for (size_t i = 0; i < lf.entries.size(); ++i) {
        std::printf("  [%zu] %s →", i + 1, lf.entries[i].text.c_str());
        for (int n : lf.entries[i].midiNotes)
            std::printf("  %s(%d)", midiToName(n).c_str(), n);
        std::printf("\n");
    }

    // Load models
    try { BinaryData::init(bundlePath); }
    catch (const std::exception& e) {
        std::fprintf(stderr, "Failed to load models: %s\n", e.what());
        return 1;
    }

    // Init test state
    TestState st;
    st.sampleRate       = static_cast<double>(wavSr);
    st.gateFloor        = rangeCfg.gateFloor;
    st.ampFloor         = rangeCfg.ampFloor;
    st.threshold        = rangeCfg.threshold;
    st.frameThreshold   = rangeCfg.frameThreshold;
    st.onsetBlankMs     = rangeCfg.onsetBlankMs;
    st.swiftF0Threshold = rangeCfg.swiftF0Threshold;
    st.mode             = rangeCfg.mode;
    st.provisionalMode  = rangeCfg.provisionalMode;
    st.octaveLockMs     = rangeCfg.octaveLockMs;
    st.maxPoly          = rangeCfg.maxPoly;
    st.bendEnabled      = rangeCfg.bendEnabled;

    // Try to load SwiftF0
    {
        const std::string bslash = (bundlePath.empty() || bundlePath.back() == '/') ? "" : "/";
        const std::string probes[] = {
            bundlePath + bslash + "swift_f0_model.onnx",
        };
        for (const auto& p : probes) {
            FILE* f = std::fopen(p.c_str(), "rb");
            if (f) {
                std::fclose(f);
                try {
                    st.swiftF0 = std::make_unique<SwiftF0Detector>(p);
                    std::printf("SwiftF0: loaded from %s\n", p.c_str());
                } catch (...) {}
                break;
            }
        }
    }

    st.pickDetector.init(static_cast<float>(st.sampleRate), 3000.0f, 3.0f);
#ifdef __aarch64__
    st.goertzel.init(static_cast<float>(st.sampleRate), NOTE_BASE, NOTE_BASE + NOTE_COUNT - 1);
#endif
    sem_init(&st.workerSem, 0, 0);

    for (const auto& rc : rangeCfg.ranges) {
        auto r             = std::make_unique<TestRangeState>();
        r->cfg             = rc;
        r->ringSize        = windowMsToRingSize(rc.windowMs);
        r->minFreshSamples = std::max(r->ringSize / 2, MIN_FRESH_FLOOR);
        r->ring.assign(RING_MAX, 0.0f);
        r->basicPitch      = std::make_unique<BasicPitch>();
        const float sr     = static_cast<float>(st.sampleRate);
        const float cutoff = std::min(midiToFreq(rc.midiHigh) * 1.2f, sr * 0.45f);
        r->obd.setLowpass(cutoff, sr);
        r->mpm.init(sr, rc.midiLow, rc.midiHigh);
        st.ranges.push_back(std::move(r));
    }

    // Print config
    std::printf("Mode:    %s\n",
                st.mode == PlayMode::MONO       ? "mono" :
                st.mode == PlayMode::SWIFT_MONO ? "swiftmono" :
                st.mode == PlayMode::SWIFT_POLY ? "swiftpoly" :
                st.mode == PlayMode::GOERTZEL_MONO ? "goertzelmono" :
                st.mode == PlayMode::GOERTZEL_POLY ? "goertzelpoly" : "poly");
    std::printf("Block:   %d samples (%.1f ms)\n", blockSize,
                blockSize * 1000.0 / st.sampleRate);
    std::printf("Ranges:  %zu\n", st.ranges.size());

    // Start worker thread
    st.startTime = std::chrono::steady_clock::now();
    st.workerThread = std::thread(testWorker, &st);

    // Process WAV in blocks (simulating JACK callback)
    std::printf("\nProcessing...\n");
    const int totalSamples = static_cast<int>(wavData.size());
    int pos = 0;
    while (pos < totalSamples) {
        const int n = std::min(blockSize, totalSamples - pos);
        processTestBlock(&st, wavData.data() + pos, n);
        pos += n;

        // Simulate real-time pacing: sleep proportional to block duration
        // This ensures the worker thread has time to process snapshots.
        const double blockDurUs = (n / st.sampleRate) * 1e6;
        std::this_thread::sleep_for(std::chrono::microseconds(
            static_cast<int64_t>(blockDurUs * 0.5)));  // 0.5× real-time is fast enough
    }

    // Wait for worker to finish remaining snapshots
    std::this_thread::sleep_for(std::chrono::milliseconds(500));

    // Shutdown worker
    st.workerQuit.store(true, std::memory_order_release);
    sem_post(&st.workerSem);
    st.workerThread.join();
    sem_destroy(&st.workerSem);

    // Drain any remaining midiOut events
    const double finalT = st.audioTimeS.load(std::memory_order_relaxed);
    for (auto& rp : st.ranges) {
        PendingNote pn;
        while (rp->midiOut.pop(pn)) {
            if (pn.type == PendingNote::PITCH_BEND) continue;
            const bool isOn = (pn.type == PendingNote::NOTE_ON);
            st.events.push_back({finalT, pn.pitch, isOn, pn.value});
        }
    }

    // Sort events by time
    std::sort(st.events.begin(), st.events.end(),
              [](const MidiEvent& a, const MidiEvent& b) { return a.timeS < b.timeS; });

    // Print all detected events
    std::printf("\n── Detected MIDI events ──────────────────────────────────────\n");
    for (const auto& e : st.events) {
        if (e.noteOn)
            std::printf("  %.3fs  ON   %-4s (%3d)  vel %d\n",
                        e.timeS, midiToName(e.midiNote).c_str(), e.midiNote, e.velocity);
        else
            std::printf("  %.3fs  OFF  %-4s (%3d)\n",
                        e.timeS, midiToName(e.midiNote).c_str(), e.midiNote);
    }

    // Score against labels
    std::printf("\n── Scoring ──────────────────────────────────────────────────\n");
    TestResult res = scoreEvents(st.events, lf);
    for (const auto& d : res.details)
        std::printf("%s\n", d.c_str());

    std::printf("\n── Summary ──────────────────────────────────────────────────\n");
    std::printf("  Entries:         %d\n", res.totalEntries);
    std::printf("  Hits:            %d  (%.0f%%)\n", res.hits,
                res.totalEntries > 0 ? 100.0 * res.hits / res.totalEntries : 0.0);
    std::printf("  Partial hits:    %d\n", res.partialHits);
    std::printf("  Misses:          %d\n", res.misses);
    std::printf("  Wrong notes:     %d\n", res.wrongNotes);
    std::printf("  Missing segments:%d\n", res.missingSegments);
    std::printf("  Extra segments:  %d\n", res.extraSegments);

    const bool pass = (res.hits == res.totalEntries
                       && res.wrongNotes == 0
                       && res.extraSegments == 0);
    std::printf("\n  Result: %s\n\n", pass ? "PASS" : "FAIL");
    return pass ? 0 : 1;
}

// ══════════════════════════════════════════════════════════════════════════════
//  MAIN
// ══════════════════════════════════════════════════════════════════════════════

static void printUsage(const char* prog)
{
    std::fprintf(stderr,
        "Usage:\n"
        "  %s record -o <output.wav> [--port <jack_port>]\n"
        "  %s test   -i <input.wav> -l <labels.txt>\n"
        "             [--bundle PATH] [--config PATH]\n"
        "             [--mode mono|poly|swiftmono|swiftpoly|goertzelmono|goertzelpoly]\n"
        "             [--max-poly N]\n"
        "             [--provisional on|swift|none|adaptive]\n"
        "             [--threshold F] [--frame-threshold F]\n"
        "             [--gate F] [--amp-floor F]\n"
        "             [--block-size N]  (JACK-like block size, default 64)\n"
        "\n", prog, prog);
}

int main(int argc, char** argv)
{
    if (argc < 2) { printUsage(argv[0]); return 1; }

    const char* cmd = argv[1];

    if (std::strcmp(cmd, "record") == 0) {
        const char* outPath  = nullptr;
        const char* jackPort = nullptr;

        for (int i = 2; i < argc; ++i) {
            if ((!std::strcmp(argv[i], "-o") || !std::strcmp(argv[i], "--output")) && i+1 < argc)
                outPath = argv[++i];
            else if (!std::strcmp(argv[i], "--port") && i+1 < argc)
                jackPort = argv[++i];
            else { printUsage(argv[0]); return 1; }
        }
        if (!outPath) { std::fprintf(stderr, "record: -o <output.wav> required\n"); return 1; }
        return runRecord(outPath, jackPort);

    } else if (std::strcmp(cmd, "test") == 0) {
        const char* inputPath = nullptr;
        const char* labelPath = nullptr;
        std::string bundlePath, configPath;
        int  blockSize = 64;
        bool modeSet = false, provSet = false;
        int  maxPolyOverride = -1;
        PlayMode mode = PlayMode::MONO;
        ProvMode prov = ProvMode::ON;
        float threshold = -1, frameThr = -1, gate = -1, ampFloor = -1;

        for (int i = 2; i < argc; ++i) {
            if ((!std::strcmp(argv[i], "-i") || !std::strcmp(argv[i], "--input")) && i+1 < argc)
                inputPath = argv[++i];
            else if ((!std::strcmp(argv[i], "-l") || !std::strcmp(argv[i], "--labels")) && i+1 < argc)
                labelPath = argv[++i];
            else if (!std::strcmp(argv[i], "--bundle") && i+1 < argc)
                bundlePath = argv[++i];
            else if (!std::strcmp(argv[i], "--config") && i+1 < argc)
                configPath = argv[++i];
            else if (!std::strcmp(argv[i], "--block-size") && i+1 < argc)
                blockSize = std::atoi(argv[++i]);
            else if (!std::strcmp(argv[i], "--threshold") && i+1 < argc)
                threshold = std::stof(argv[++i]);
            else if (!std::strcmp(argv[i], "--frame-threshold") && i+1 < argc)
                frameThr = std::stof(argv[++i]);
            else if (!std::strcmp(argv[i], "--gate") && i+1 < argc)
                gate = std::stof(argv[++i]);
            else if (!std::strcmp(argv[i], "--amp-floor") && i+1 < argc)
                ampFloor = std::stof(argv[++i]);
            else if (!std::strcmp(argv[i], "--mode") && i+1 < argc) {
                const char* s = argv[++i];
                if      (!std::strcmp(s, "mono"))         mode = PlayMode::MONO;
                else if (!std::strcmp(s, "swiftmono"))    mode = PlayMode::SWIFT_MONO;
                else if (!std::strcmp(s, "swiftpoly"))    mode = PlayMode::SWIFT_POLY;
                else if (!std::strcmp(s, "goertzelmono")) mode = PlayMode::GOERTZEL_MONO;
                else if (!std::strcmp(s, "goertzelpoly")) mode = PlayMode::GOERTZEL_POLY;
                else                                      mode = PlayMode::POLY;
                modeSet = true;
            }
            else if (!std::strcmp(argv[i], "--provisional") && i+1 < argc) {
                const char* s = argv[++i];
                if      (!std::strcmp(s, "swift"))    prov = ProvMode::SWIFT;
                else if (!std::strcmp(s, "adaptive")) prov = ProvMode::ADAPTIVE;
                else if (!std::strcmp(s, "none") || !std::strcmp(s, "off"))
                                                      prov = ProvMode::NONE;
                else                                  prov = ProvMode::ON;
                provSet = true;
            }
            else if (!std::strcmp(argv[i], "--max-poly") && i+1 < argc)
                maxPolyOverride = std::max(1, std::min(6, std::atoi(argv[++i])));
            else { printUsage(argv[0]); return 1; }
        }

        if (!inputPath || !labelPath) {
            std::fprintf(stderr, "test: -i <input.wav> and -l <labels.txt> required\n");
            return 1;
        }

        // Discover bundle path
        if (bundlePath.empty()) {
            std::string selfDir;
            if (argc > 0) {
                std::string self(argv[0]);
                auto slash = self.rfind('/');
                if (slash != std::string::npos) selfDir = self.substr(0, slash + 1);
            }
            const std::string probes[][2] = {
                { selfDir, selfDir + "ModelData/cnn_contour_model.json" },
                { selfDir + "pipitch.lv2",
                  selfDir + "pipitch.lv2/ModelData/cnn_contour_model.json" },
                { "/zynthian/zynthian-plugins/lv2/pipitch.lv2",
                  "/zynthian/zynthian-plugins/lv2/pipitch.lv2/ModelData/cnn_contour_model.json" },
            };
            for (const auto& p : probes) {
                FILE* f = std::fopen(p[1].c_str(), "rb");
                if (f) { std::fclose(f); bundlePath = p[0]; break; }
            }
            if (bundlePath.empty()) {
                std::fprintf(stderr, "Cannot find model files. Use --bundle <path>\n");
                return 1;
            }
        }

        // Load config
        RangeConfig rangeCfg;
        if (!configPath.empty()) {
            rangeCfg = loadRangeConfig(configPath);
        } else {
            // Try default config locations
            std::string selfDir;
            if (argc > 0) {
                std::string self(argv[0]);
                auto slash = self.rfind('/');
                if (slash != std::string::npos) selfDir = self.substr(0, slash + 1);
            }
            const std::string probes[] = {
                selfDir + "pipitch_tune.conf",
                bundlePath + "/pipitch_tune.conf",
            };
            for (const auto& p : probes) {
                FILE* f = std::fopen(p.c_str(), "r");
                if (f) { std::fclose(f); rangeCfg = loadRangeConfig(p); configPath = p; break; }
            }
        }

        // CLI overrides
        if (threshold >= 0)  rangeCfg.threshold      = threshold;
        if (frameThr >= 0)   rangeCfg.frameThreshold  = frameThr;
        if (gate >= 0)       rangeCfg.gateFloor       = gate;
        if (ampFloor >= 0)   rangeCfg.ampFloor         = ampFloor;
        if (modeSet)         rangeCfg.mode             = mode;
        if (provSet)         rangeCfg.provisionalMode  = prov;
        if (maxPolyOverride >= 1) rangeCfg.maxPoly    = maxPolyOverride;

        if (rangeCfg.ranges.empty()) {
            NoteRange low;  low.name = "low";  low.midiLow = 0;  low.midiHigh = 48;
            low.windowMs = 150; low.holdCycles = 2;
            rangeCfg.ranges.push_back(low);

            NoteRange high; high.name = "high"; high.midiLow = 49; high.midiHigh = 127;
            high.windowMs = 100; high.holdCycles = 0;
            rangeCfg.ranges.push_back(high);
        }

        return runTest(inputPath, labelPath, bundlePath, configPath, rangeCfg, blockSize);

    } else {
        printUsage(argv[0]);
        return 1;
    }
}
