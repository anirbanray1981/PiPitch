# NeuralNote — Claude Code Context

> **This file is git-ignored** (listed in `.gitignore`).
> It exists solely to survive conversation compaction and give Claude full context on the next session.

---

## Project Goal

Convert a guitar's audio signal to MIDI in real time on a Raspberry Pi 5 (Zynthian), with the lowest possible perceived latency. The plugin runs as an LV2 unit inside Zynthian's JACK graph.

The system uses **two-phase pitch detection**:
1. **Fast provisional note** — fires within ~15–50 ms of a pick attack using OBP + HPS + MPM.
2. **CNN confirmation/correction** — BasicPitch (RTNeural) runs in a background thread (~95 ms on Pi 5) and either confirms, corrects, or cancels the provisional.

---

## Pi 5 Access

| Field | Value |
|---|---|
| Host | `10.141.1.206` |
| User | `root` |
| Password | `opensynth` |
| SSH command | `sshpass -p opensynth ssh root@10.141.1.206` |
| SCP command | `sshpass -p opensynth scp <file> root@10.141.1.206:<dest>` |

---

## Repository Paths

| Location | Path |
|---|---|
| Local source (this repo) | `/home/patch/NeuralNote/` |
| Pi 5 source mirror | `/root/neuralnote_src/` |
| Pi 5 build directory | `/root/neuralnote_build/` |
| LV2 bundle (deployed) | `/zynthian/zynthian-plugins/lv2/neuralnote_guitar2midi.lv2/` |

**Workflow:** Edit locally → `scp` changed files to `/root/neuralnote_src/LV2/` → compile on Pi 5 → test with JACK.

**cmake cannot be re-run on Pi 5** (JUCE submodule is absent). Compile targets manually using the flags extracted from `CMakeFiles/<target>.dir/flags.make` and relink via `bash CMakeFiles/<target>.dir/link.txt`.

---

## Build Targets

### LV2 plugin (two shared libraries loaded at runtime by the wrapper)

| Target | Output `.so` | `-march` flag | Notes |
|---|---|---|---|
| `neuralnote_impl_neon` | `neuralnote_impl_neon.so` | `armv8-a+simd` | Pi 4 (Cortex-A72). No MPM / no FFTW. |
| `neuralnote_impl_armv82` | `neuralnote_impl_armv82.so` | `armv8.2-a+dotprod+fp16` | **Pi 5** (Cortex-A76). MPM enabled. Links `libfftw3f`. |

Both live in `neuralnote_build/neuralnote_guitar2midi.lv2/`.
The wrapper `neuralnote_guitar2midi.so` dlopen-selects the right impl at runtime via `AT_HWCAP`.

Pi 5 targets use `-mcpu=cortex-a76 -mtune=cortex-a76`. The neon (Pi 4) target retains `-mcpu=cortex-a72`.

#### Compile a single impl `.o` (example: armv82):
```bash
cd /root/neuralnote_build
DEFINES="-DNEURALNOTE_IMPL_NAME=\"neuralnote_impl_armv82\" \
  -DRTNEURAL_DEFAULT_ALIGNMENT=16 -DRTNEURAL_NAMESPACE=RTNeural \
  -DRTNEURAL_USE_EIGEN=1 -DSAVE_DOWNSAMPLED_AUDIO=0 \
  -DUSE_TEST_NOTE_FRAME_TO_TIME=0 -Dneuralnote_impl_armv82_EXPORTS"
INCLUDES="-I/root/neuralnote_src/LV2 -I/root/neuralnote_src/NeuralNote/Lib/Model \
  -I/root/neuralnote_src/NeuralNote/Lib/Utils \
  -I/root/neuralnote_src/NeuralNote/ThirdParty/RTNeural \
  -I/root/neuralnote_src/NeuralNote/ThirdParty/onnxruntime/include \
  -I/root/neuralnote_src/NeuralNote/ThirdParty/RTNeural/RTNeural/../modules/json \
  -I/root/neuralnote_src/NeuralNote/ThirdParty/RTNeural/RTNeural/.. \
  -I/root/neuralnote_src/NeuralNote/ThirdParty/RTNeural/modules/Eigen"
/usr/bin/c++ $DEFINES $INCLUDES \
  -mcpu=cortex-a76 -mtune=cortex-a76 -O3 -DNDEBUG -fPIC -O2 -fPIC \
  -march=armv8.2-a+dotprod+fp16 -std=gnu++17 \
  -c /root/neuralnote_src/LV2/neuralnote_impl.cpp \
  -o CMakeFiles/neuralnote_impl_armv82.dir/LV2/neuralnote_impl.cpp.o
bash CMakeFiles/neuralnote_impl_armv82.dir/link.txt
```

### Tuning / latency tool (`neuralnote_tune`)

Standalone JACK program. Mirrors the plugin logic with extra console logging. Used for live tuning and testing.

Run directly from the build directory — no install step needed:
```bash
/root/neuralnote_build/neuralnote_tune --config neuralnote_tune.conf
```
No `LD_LIBRARY_PATH` required: `libonnxruntime` is resolved via rpath baked in at link time; `libfftw3f` is statically linked.

```bash
# On Pi 5 — build only tune target:
cd /root/neuralnote_build
DEFINES="-DRTNEURAL_DEFAULT_ALIGNMENT=16 -DRTNEURAL_NAMESPACE=RTNeural \
  -DRTNEURAL_USE_EIGEN=1 -DSAVE_DOWNSAMPLED_AUDIO=0 -DUSE_TEST_NOTE_FRAME_TO_TIME=0"
INCLUDES="-I/root/neuralnote_src/LV2 -I/root/neuralnote_src/NeuralNote/Lib/Model \
  -I/root/neuralnote_src/NeuralNote/Lib/Utils \
  -I/root/neuralnote_src/NeuralNote/ThirdParty/RTNeural \
  -I/root/neuralnote_src/NeuralNote/ThirdParty/onnxruntime/include \
  -I/root/neuralnote_src/NeuralNote/ThirdParty/RTNeural/RTNeural/../modules/json \
  -I/root/neuralnote_src/NeuralNote/ThirdParty/RTNeural/RTNeural/.. \
  -I/root/neuralnote_src/NeuralNote/ThirdParty/RTNeural/modules/Eigen"
/usr/bin/c++ $DEFINES $INCLUDES \
  -mcpu=cortex-a76 -mtune=cortex-a76 -O3 -DNDEBUG -fPIE -O2 -std=gnu++17 \
  -c /root/neuralnote_src/LV2/neuralnote_tune.cpp \
  -o CMakeFiles/neuralnote_tune.dir/LV2/neuralnote_tune.cpp.o
bash CMakeFiles/neuralnote_tune.dir/link.txt
```

---

## Key Source Files

| File | Purpose |
|---|---|
| `LV2/neuralnote_impl.cpp` | LV2 plugin — the production audio path |
| `LV2/neuralnote_tune.cpp` | JACK tuning tool — mirrors impl with logging |
| `LV2/NeuralNoteShared.h` | Shared constants, structs, and pipeline helpers (see below) |
| `LV2/OneBitPitchDetector.h` | OBP: 4th-order Butterworth → adaptive Schmitt → period averaging |
| `LV2/McLeodPitchDetector.h` | MPM: FFT autocorrelation → NSDF → parabolic interpolation |
| `LV2/NoteRangeConfig.h` | Config structs (`NoteRange`, `RangeConfig`, `PlayMode`) and `.conf` parser |
| `LV2/neuralnote_ranges.conf` | LV2 bundle range config (range sections only — no global keys) |
| `LV2/neuralnote_tune.conf` | Tune tool config (includes global keys; shipped in the repo) |
| `LV2/plugin.ttl` | LV2 port definitions and defaults |
| `LV2/SwiftF0Detector.h` | SwiftF0 ONNX wrapper: `infer()` → median-confident MIDI note or -1 |
| `LV2/swift_f0_model.onnx` | SwiftF0 model (389 KB, 95K params, 16 kHz, 16 ms/frame) |
| `LV2/neuralnote-connect.sh` | JACK MIDI fan-out script (ch0_out → extra synth chains) |
| `LV2/neuralnote-connect.service` | systemd oneshot unit that runs fan-out after zynthian.service |

### NeuralNoteShared.h contents

Everything shared between `neuralnote_impl.cpp` and `neuralnote_tune.cpp`:

| Symbol | Kind | Purpose |
|---|---|---|
| `PLUGIN_SR`, `NOTE_BASE/COUNT`, `RING_MAX`, etc. | constants | Shared numerical limits |
| `PickDetector` | struct | HPF-based pick onset detector: 3 kHz 1st-order IIR HPF → dual-EMA (fast 0.1 ms / slow 20 ms) → fast/slow ratio fires onset; two-tier: ≥10 immediate, 4–10 needs RMS confirmation |
| `PendingNote` | struct | `{bool noteOn; int pitch; int velocity}` (velocity 0–127) |
| `MidiOutQueue` | struct | SPSC ring buffer, worker → audio thread |
| `SnapshotChannel` | struct | SPSC audio → worker (atomic + semaphore); includes `onsetDispatched` flag |
| `RangeStateBase` | struct | All per-range fields: `basicPitch`, `ring`, `midiOut`, OBP/HPS/MPM state, note bitmaps, `hasActiveNotes`, `activeNotesBits`, `provLastSeenByCNN`, `provCancelGrace`, `swiftOnsetGrace`, `swiftGraceStaleNote`, `provCooldownRemain`, `provCooldownNote` |
| `pushRingSamples` | template fn | Write pre-resampled samples into circular ring |
| `armOrExpireOBP` | template fn | Arm OBP on onset (100 ms window); decrement countdown |
| `resetOBPOnGate` | template fn | Clear OBP/HPS/MPM on gated silence |
| `runOBPHPS` | template fn | OBP sub-block loop + Layer 1 harmonic + Layer 2 HPS → MIDI note or -1 |
| `dispatchSnapshotIfReady` | template fn | Linearise ring → snapshot slot, sem_post; includes RMS gate and hasActiveNotes bypass |
| `buildNNBits` | fn | CNN note events → `newBits`/`newVel` arrays |
| `applyNotesDiff` | fn | Core note ON/OFF/hold state machine; returning notes cancel hold (no OFF+ON); updates `hasActiveNotes` + `activeNotesBits` |

**`NEURALNOTE_ENABLE_MPM`** must be `#define`d **before** `#include "NeuralNoteShared.h"` to compile in McLeod call-sites. impl.cpp defines it conditionally on `__ARM_FEATURE_DOTPROD`; tune.cpp always defines it.

---

## Architecture

### Per-range pipeline (one instance per `[range]` in the config)

```
JACK callback (RT thread)
  │
  ├─ Onset detection (two-tier):
  │    Primary: PickDetector (HPF 3 kHz → dual-EMA fast/slow ratio)
  │      Tier 1: ratio ≥ 10 → immediate onset (sub-buffer sample precision)
  │      Tier 2: ratio 4–10 → confirmed only if RMS also exceeds threshold
  │    Fallback: RMS onset (blockRms > smoothedRms × 3.0) for hammer-ons etc.
  │    → arms OBP + MPM windows, force-dispatches rings, sets onsetDispatched
  │
  ├─ OBP (OneBitPitchDetector)
  │    4th-order Butterworth LP → adaptive Schmitt trigger → period averaging
  │    Voting buffer: N_CONSEC=4 consecutive readings must agree
  │    Note cap: OBP_NOTE_CAP=76 (E5) — OBP above this is rejected
  │
  ├─ HPS (Bit-parallel Harmonic Product Spectrum)
  │    Accumulates all OBP detections in obpHpsBits per range
  │    Cross-range OR → shift >>12 (÷2 Hz) and >>19 (÷3 Hz) → AND
  │    Lowest surviving bit = true fundamental
  │
  ├─ MPM (McLeodPitchDetector)  [Pi 5 / armv82 only]
  │    FFT-based autocorrelation (FFTW3f, 4096-pt) → NSDF
  │    Two-pass peak picking → parabolic interpolation
  │    Runs on native-SR audio accumulated since onset
  │
  │  Provisional fires only if OBP + HPS + MPM agree on the same MIDI note
  │  (Pi 4: OBP + HPS only — MPM not compiled in)
  │
  ├─ Ring buffer fill (22050 Hz resampled audio for CNN)
  │
  └─ Snapshot dispatch → worker thread (semaphore)
       │
       ├─ [mode=0/1] BasicPitch CNN (RTNeural)  ~95 ms on Pi 5
       │    └─ buildNNBits() + applyNotesDiff() → confirms / corrects / cancels provisional
       │         cancelledProv flag → immediate OFF (bypasses holdCycles)
       │         obdBlacklistNote → suppresses same wrong note on next onset
       │
       └─ [mode=2 swiftmono] SwiftF0 (ONNX)  ~10–20 ms on Pi 5
            resample 22050→16 kHz inline
            SwiftF0Detector::infer() → median confident Hz → MIDI note
            → single-bit newBits → applyNotesDiff() (mono=true)
```

### Per-range struct hierarchy

```
RangeStateBase   (NeuralNoteShared.h)
  ├── RangeState (neuralnote_impl.cpp) — no additional fields (empty derived struct)
  └── RangeState (neuralnote_tune.cpp) — adds provMidiPitch, provOnTimeMs
```

Both derived structs are named `RangeState`; they are distinct types in separate translation units.

### Global parameters — impl.cpp

`threshold`, `frame_threshold`, `amp_floor`, `gate_floor`, `mode`, `onset_blank_ms` are plugin-level ports on `NeuralNotePlugin`:

```
LV2 host → port buffers → run() (every callback, no change-detection)
         → thresholdVal / frameThresholdVal / ampFloorVal / modeVal atomics
         → worker reads them before each inference
```

Port values are propagated unconditionally every `run()` so that preset/snapshot restoration is always honoured.

In multi-range mode (conf file present), global defaults come from the LV2 port defaults in `plugin.ttl` — **not** from `neuralnote_ranges.conf` (which contains range sections only).

### Worker thread flow (impl.cpp)

```
BasicPitch mode: CNN result → buildNNBits() → cancel grace → applyNotesDiff()
SwiftF0 mode:    resample inline → SwiftF0::infer() → single-bit newBits
                 → onset-transition grace → decay-tail ghost suppression
                 → applyNotesDiff(holdCyclesOverride=swiftHoldCycles)
                 (cancel grace skipped in swiftmono)
```

### Worker thread flow (tune.cpp)

```
BasicPitch mode: CNN result → buildNNBits() [shared]
                            → cancel grace
                            → applyRangeDiff() [local wrapper]:
                                CNN outcome logging
                                → applyNotesDiff() [shared]
                                → ON/OFF logging

SwiftF0 mode:    resample inline → SwiftF0::infer() → single-bit newBits
                 → onset-transition grace (suppress note changes 1 cycle after onset)
                 → decay-tail ghost suppression (no new notes >250 ms after onset)
                 → "SwiftF0 → <note>" log (deduplicated per range)
                 → cancel grace skipped
                 → applyNotesDiff(holdCyclesOverride=swiftHoldCycles) [shared]
                 → ON/OFF logging
```

### Threading

- **JACK callback** (RT thread): onset detection, OBP, HPS, MPM, ring fill, dispatch.
- **Worker thread** (one, shared across all ranges): CNN inference, `buildNNBits`, `applyNotesDiff`, MIDI output.
- All comms are lockless: `SnapshotChannel` (SPSC atomic + semaphore), `MidiOutQueue` (SPSC ring).

---

## LV2 Port Layout (plugin.ttl)

| Index | Symbol | Default | Range | Notes |
|---|---|---|---|---|
| 0 | `audio_in` | — | — | Audio input |
| 1 | `midi_out` | — | — | MIDI output (atom sequence) |
| 2 | `audio_out` | — | — | Audio through |
| 3 | `threshold` | 0.6 | 0.1–1.0 | Onset sensitivity |
| 4 | `gate_floor` | 0.003 | 0.0–0.1 | Noise gate floor |
| 5 | `amp_floor` | 0.65 | 0.0–1.0 | BasicPitch amplitude floor |
| 6 | `frame_threshold` | 0.5 | 0.05–0.95 | Per-frame CNN confidence |
| 7 | `mode` | 1 (mono) | 0=poly / 1=mono / 2=swiftmono | Polyphony mode |
| 8 | `onset_blank_ms` | 25 | 10–100 ms | Re-trigger suppression window |

---

## Key Constants & Parameters

| Symbol | Value | Meaning |
|---|---|---|
| `NOTE_BASE` | 40 | Lowest MIDI note in bitmap (E2) |
| `NOTE_COUNT` | 49 | Bitmap covers MIDI 40–88 (E2–E6) |
| `OBP_NOTE_CAP` | 76 | OBP provisionals above E5 rejected |
| `OBP_CHUNK` | 16 | Sub-block size for mid-buffer OBP firing |
| `N_CONSEC` (OBPVotingBuffer) | 4 | Consecutive agreeing OBP readings required before provisional fires |
| `HYST_RATIO` (OBP) | 0.25 | Schmitt threshold = 0.25 × filtered amplitude EMA |
| `obdWindowRemain` | 100 ms | Fixed OBP detection window per onset |
| `MPM_BUFSIZE` | 2048 | Max MPM analysis window (native SR samples) |
| `MPM_FFTSIZE` | 4096 | Zero-padded FFT size (≥ 2×BUFSIZE−1) |
| `MPM_K` | 0.86 | NSDF key-maximum threshold |
| `ONSET_RATIO` | 3.0× | RMS fallback: block RMS > 3× smoothed background |
| `ONSET_BLANK_MS` | 25 ms | Re-trigger suppression fallback constant (overridden by port 8 in LV2; by `onset_blank_ms` conf key in tune) |
| `PICK_HIGH_TIER` | 10.0 | PickDetector fast/slow ratio for immediate onset (tier 1) |
| PickDetector HPF cutoff | 3000 Hz | 1st-order IIR high-pass isolates pick "snap" |
| PickDetector fast EMA | 0.1 ms | Tracks transients within ~0.3 ms |
| PickDetector slow EMA | 20 ms | Smooths harmonic oscillations; baseline for ratio |
| PickDetector slopeRatio | 4.0 | Minimum fast/slow ratio to fire (tier 2; tier 1 = 10.0) |
| `ONSET_GATE_S` | 0.25 s | Decay-tail ghost suppression: max time after onset for SwiftF0 new note-ONs |
| Provisional cooldown | 200 ms | Same-note re-trigger suppression after a provisional fires |
| `MIN_FRESH_FLOOR` | 25 ms | Minimum fresh audio before CNN dispatch |
| `PLUGIN_SR` / `AUDIO_SAMPLE_RATE` | 22050 Hz | Resampled rate fed to BasicPitch CNN |

### Config file fields

**Global keys** (neuralnote_tune.conf only — not used in the LV2 plugin conf):
`gate_floor`, `amp_floor`, `threshold`, `frame_threshold`, `mode` (poly/mono/swiftmono), `onset_blank_ms`, `swift_threshold`

**Per-range keys** (both neuralnote_tune.conf and neuralnote_ranges.conf):
`name`, `midi_low`, `midi_high`, `window` (ms), `min_note_length` (CNN frames ≈ 11.6 ms), `hold_cycles`, `swift_hold_cycles`

---

## SwiftF0 Integration — `swiftmono` mode

### Overview

`swiftmono` adds a third polyphony mode (mode=2) that replaces BasicPitch CNN confirmation with SwiftF0, a 389 KB monophonic pitch estimator. OBP+HPS+MPM provisional detection is unchanged. Only the worker-thread inference layer changes.

**Model**: `swift_f0_model.onnx` in the LV2 bundle root (not ModelData/).
**Paper**: arxiv.org/html/2508.18440v1 — 95K parameters, 16 kHz input, 16 ms frame (256-sample hop), outputs `pitch_hz[]` + `confidence[]`.
**Confirmed Pi 5 inference**: model loads successfully; estimated ~10–20 ms per window (vs ~95 ms for BasicPitch).
**Tensor names**: input=`input_audio` shape `(1, N)` float32; outputs=`pitch_hz`, `confidence` shape `(1, F)` float32.

### New files

| File | Purpose |
|---|---|
| `LV2/SwiftF0Detector.h` | ONNX Runtime C++ wrapper: `Ort::Session` owner; `infer()` returns single MIDI note |
| `LV2/swift_f0_model.onnx` | SwiftF0 ONNX model (389 KB) — lives in bundle root, copied by CMake POST_BUILD |

### Changes to existing files

| File | Change |
|---|---|
| `NoteRangeConfig.h` | `PlayMode::SWIFT_MONO` enum value; `swiftF0Threshold = 0.5f` in `RangeConfig`; parse `swift_threshold` and `mode = swiftmono` conf keys |
| `plugin.ttl` | Port 7 `mode` max 1→2; new scale point `SwiftMono = 2` |
| `neuralnote_impl.cpp` | Load `swift_f0_model.onnx` at instantiate (graceful fail if absent); worker branches on `modeVal == 2`: resample 22050→16 kHz inline, call `SwiftF0Detector::infer()`, build single-note `newBits`; LV2 uses hardcoded threshold=0.5f |
| `neuralnote_tune.cpp` | Same worker branch; `Monitor::swiftF0` + `sf0Buf` + `swiftF0Threshold`; loads model from binary dir or bundle; `--swift-threshold` CLI flag; `swift_threshold` conf key; SwiftF0-specific log lines |
| `neuralnote_tune.conf` | `swift_threshold = 0.5` added as global key |
| `CMakeLists.txt` | POST_BUILD copies `LV2/swift_f0_model.onnx` → bundle |

### Worker thread logic (swiftmono path)

```
snapshot dispatched (same per-range ring mechanism as before)
  │
  ├─ Resample inline: 22050 Hz → 16000 Hz (linear interpolation, ratio 0.7256)
  │    22050 Hz ring buffer → sf0Buf (worker-only scratch vector)
  │    snapChan.ready.store(false) after copy — unblocks audio thread
  │
  ├─ SwiftF0Detector::infer(sf0Buf, nSamples16k, threshold=0.5)
  │    → collects frames where confidence ≥ threshold and pitchHz > 20 Hz
  │    → takes median of confident frame pitches
  │    → Hz → MIDI (69 + 12 × log2(Hz / 440)) → rounded int
  │    → returns -1 if no confident frames
  │
  ├─ Range filter: if MIDI note ∈ [midiLow, midiHigh] and ∈ [NOTE_BASE, NOTE_BASE+COUNT)
  │    → set single bit in newBits, newVel[bit] = 100
  │
  ├─ Cancel grace (same as BasicPitch path — provCancelGrace suppresses first-cycle cancel)
  │
  └─ applyNotesDiff(r, newBits, newVel, provForDiff, mono=true)
       [unchanged state machine — hold, blacklist, cross-range kill all apply]
```

### SwiftF0Detector.h design

- Header-only class; `#include "onnxruntime_cxx_api.h"` (in `ThirdParty/onnxruntime/include/`)
- Constructor takes `modelPath` (full path string); throws on failure
- `Ort::Env`, `Ort::SessionOptions`, `Ort::Session` owned as members; single-threaded inference (`SetIntraOpNumThreads(1)`)
- `infer(const float* audio16k, int nSamples, float threshold)`:
  - Creates `(1, nSamples)` input tensor pointing at caller's buffer (const_cast, ORT won't modify)
  - Runs session: inputNames=`{"input_audio"}`, outputNames=`{"pitch_hz", "confidence"}`
  - Reads `(1, F)` outputs, collects Hz where conf ≥ threshold and Hz > 20
  - Returns median Hz → MIDI note (rounded int), or -1 if no confident frames
- Owned at plugin/monitor level — one instance shared across all ranges

### Cancel grace in swiftmono

`provCancelGrace` applies identically. At ~10–20 ms inference the first dispatch arrives faster than BasicPitch, but the grace cycle is harmless and suppresses the same legato-retrigger edge case.

### Key parameters

| Parameter | Source | Default |
|---|---|---|
| `swift_threshold` | `neuralnote_tune.conf` / `--swift-threshold` CLI | 0.5 |
| LV2 swift threshold | hardcoded in impl.cpp worker | 0.5f |
| `amp_floor` | LV2 port 5 / conf key | 0.65 (not used in swiftmono path) |
| `frame_threshold` | LV2 port 6 | not used in swiftmono path |

**Confidence calibration**: pure sine A4 → 0.72–0.80 confidence; pure sine E2 → 0.90–0.96. Guitar notes vary. Default 0.5 is permissive — raise to 0.7–0.8 if false detections occur.

### Resampling

Inlined in worker (not a shared helper). Ratio = 16000 / 22050 ≈ 0.7256. Linear interpolation over `snapChan.data` → `sf0Buf`. For 80 ms ring: 1764 → 1280 samples. Negligible cost (~1 µs).

---

## NoteRangeConfig.h — struct layout

```cpp
enum class PlayMode { POLY, MONO, SWIFT_MONO };

struct NoteRange {           // per-range (from [range] sections)
    std::string name;
    int   midiLow, midiHigh;
    float windowMs;          // inference window in ms
    int   minNoteLength;     // CNN frames
    int   holdCycles;
    int   swiftHoldCycles;    // hold cycles for SwiftF0 (faster cycle time)
};

struct RangeConfig {         // global
    std::vector<NoteRange> ranges;
    float    gateFloor        = 0.003f;
    float    ampFloor         = 0.65f;
    float    threshold        = 0.6f;
    float    frameThreshold   = 0.5f;
    float    onsetBlankMs     = 25.0f;
    float    swiftF0Threshold = 0.5f;  // SwiftF0 per-frame confidence threshold
    PlayMode mode             = PlayMode::MONO;
};
```

---

## Tune.cpp-specific features (not in impl.cpp)

- **Console logging**: all note events printed with timestamps, including CNN outcome (confirmed/corrected/cancelled) and MPM result (agree/disagree/not-ready with buffer fill level).
- **Synth event logging (`>>`)**: `processSynth` drain loop prints every note-ON and note-OFF actually sent to the synth engine, prefixed with `>>`. Distinguishes synth-level events from higher-level `ON`/`OFF`/`--` diagnostic lines.
- **Synth engine**: simple ADSR sine/saw/square for audio feedback.
- **MPM diagnostic prints**: every OBP vote prints `OBP+MPM agree (fill N)`, `MPM disagrees`, or `MPM not ready (fill N/2048)`. If OBP window expires without a vote, prints `OBP window expired`.
- **Monitor struct**: carries `threshold`, `frameThreshold`, `mode`, `onsetBlankMs`, `gateFloor`, `swiftF0Threshold`, `swiftF0` (unique_ptr), `sf0Buf` (worker scratch). Worker calls `r.basicPitch->setParameters(...)` in both modes (no-op in swiftmono but keeps state clean).
- **CLI**: `--threshold`, `--frame-threshold`, `--mode mono|poly|swiftmono`, `--swift-threshold`, `--gate`, `--amp-floor`, `--window`, `--hold-cycles`, `--onset-blank`, `--waveform`, `--attack`, `--release`, `--volume`. CLI values override conf file.
- **SwiftF0 model discovery**: tune looks for `swift_f0_model.onnx` next to the binary first, then in the bundle directory. Graceful degradation if not found — swiftmono falls back to BasicPitch.
- **SwiftF0 log line**: `[+T]  --   SwiftF0 → <note> (<midi>)  [inf Xms  range Y]` or `SwiftF0 → silent`.

## MPM behaviour notes

- MPM is compiled into `neuralnote_tune` unconditionally (`#define NEURALNOTE_ENABLE_MPM 1` before includes).
- MPM is compiled into `neuralnote_impl_armv82` only (`__ARM_FEATURE_DOTPROD` defined on Pi 5 Cortex-A76).
- `libfftw3f` is **statically linked** into the tune binary (not visible in `ldd`).
- `libonnxruntime.so.1` resolves via rpath to `/root/neuralnote_src/NeuralNote/ThirdParty/onnxruntime/lib-linux-aarch64/` — no `LD_LIBRARY_PATH` needed.

---

## Deployment

```bash
sshpass -p opensynth ssh root@10.141.1.206 '
cp /root/neuralnote_src/LV2/plugin.ttl              /root/neuralnote_build/neuralnote_guitar2midi.lv2/plugin.ttl
cp /root/neuralnote_src/LV2/neuralnote_ranges.conf  /root/neuralnote_build/neuralnote_guitar2midi.lv2/neuralnote_ranges.conf
cp /root/neuralnote_src/LV2/neuralnote_tune.conf    /root/neuralnote_build/neuralnote_guitar2midi.lv2/neuralnote_tune.conf
cp /root/neuralnote_src/LV2/swift_f0_model.onnx     /root/neuralnote_build/neuralnote_guitar2midi.lv2/swift_f0_model.onnx
cp -r /root/neuralnote_build/neuralnote_guitar2midi.lv2/* /zynthian/zynthian-plugins/lv2/neuralnote_guitar2midi.lv2/
systemctl restart zynthian'
```

> **Always copy `plugin.ttl`, `neuralnote_ranges.conf`, `neuralnote_tune.conf`, and `swift_f0_model.onnx` into the build bundle before deploying** — cmake does not rebuild these files, so they must be copied manually from `/root/neuralnote_src/LV2/` each time.

### `make install` (full deploy)

```bash
cd /root/neuralnote_build
cmake --install .
```

Deploys bundle to `LV2_INSTALL_PATH` (`/zynthian/zynthian-plugins/lv2`), installs `neuralnote_tune` and `neuralnote-connect.sh` to `/usr/local/bin/`, installs and conditionally enables `neuralnote-connect.service`, then restarts Zynthian and the service.

### MIDI fan-out — multiple synth chains

`neuralnote-connect.sh` wires `ZynMidiRouter:ch0_out` to additional synth chains that Zynthian's autoconnect doesn't manage. Edit `EXTRA_DSTS` in the script to match your layout. The systemd service runs it at boot after `zynthian.service`.

For all chains to receive MIDI simultaneously, `SINGLE_ACTIVE_CHANNEL` must be `0` in the Zynthian MIDI profile:

```bash
# /zynthian/config/midi-profiles/default.sh
export ZYNTHIAN_MIDI_SINGLE_ACTIVE_CHANNEL="0"
```

Each synth chain must also be set to **MIDI channel 1** (0-indexed channel 0) to match `ch0_out`.

---

---

## History of Key Fixes

1. **HYST_RATIO 0.5 → 0.25** — Schmitt trigger at 50% caused 2nd harmonic bias → wrong note. Fixed.
2. **OBP note cap E5** — Reject OBP provisionals above MIDI 76.
3. **Bit-parallel HPS** — Cross-range OBP registers ORed then shifted ×2/×3 to find true fundamental.
4. **One-onset blacklist** — `obdBlacklistNote`: CNN-cancelled note suppressed on very next onset.
5. **cancelledProv bypass** — Cancelled provisional bypasses `holdCycles`; force-expire if in hold.
6. **Out-of-bitmap prov** — Always clear `provNoteAtDispatch`; send immediate OFF for out-of-bitmap notes.
7. **MPM (McLeodPitchDetector)** — FFT autocorrelation + NSDF + parabolic interpolation; three-way OBP+HPS+MPM consensus. Pi 5 only.
8. **NeuralNoteShared.h refactor** — All shared code extracted: constants, structs, pipeline template functions, `buildNNBits`, `applyNotesDiff`.
9. **OBP minNoteLength gate removed** — OBP fires as soon as N_CONSEC=4 readings agree; no minimum-duration wait.
10. **MPM diagnostic prints** — tune.cpp prints agree/disagree/not-ready on every OBP vote; "OBP window expired" on timeout.
11. **threshold/frame_threshold/mode made global** — Moved from per-range `NoteRange` to global `RangeConfig`. Exposed as LV2 ports; tune.cpp reads from `Monitor` struct. `neuralnote_ranges.conf` contains range sections only.
12. **`-mcpu=cortex-a72` → `-mcpu=cortex-a76`** — All Pi 5 build targets updated (neon target retains A72).
13. **neuralnote_monitor removed** — Source file deleted; CMakeLists.txt build rules removed.
14. **neuralnote_ranges.conf added to bundle** — Default 4-range config (E2-B2, C3-B3, C4-B4, C5+) shipped with LV2 bundle. No global keys — those come from plugin.ttl defaults.
15. **Preset/snapshot correctness** — run() propagates all port values to worker atomics unconditionally every callback (no change-detection). Host-restored preset values are always honoured.
16. **CNN staleness check** — Worker checks `r.provNote.load() == provNoteAtDispatch` before applying provisional logic. If a new provisional fired since dispatch, `provForDiff = -1`: the stale CNN result cannot cancel the new provisional. `provNote` is left intact.
17. **Mono mode** — `applyNotesDiff` takes `bool mono`; reduces `newBits` to single highest-velocity note and updates `monoHeldNote` atomic after state machine. Worker cross-range kill: when new note-ONs appear in any range, all other ranges get immediate note-offs and `activeNotes` cleared. Audio-thread provisional kill: reads `monoHeldNote` on all ranges before firing a new provisional; sends note-offs via forge/synth voices for any held note != new note. tune.cpp `processSynth` releases all voices to state=3 (release) when new provisional fires in mono mode.
18. **MPM pending retry** — When OBP votes but MPM hasn't accumulated enough samples, save vote in `obdPendingNote` instead of discarding. Retry `mpm.analyze()` each subsequent callback; fire provisional when MPM confirms. 100 ms timeout. Clears on new onset or gate.
19. **Double-drain midiOut** — Second drain of all `midiOut` queues at the end of `run()` (after audio processing) catches CNN events pushed during the current callback, reducing CNN note latency by up to one JACK buffer.
20. **RMS gate before CNN dispatch** — `dispatchSnapshotIfReady` computes ring buffer mean-square before copying; skips `sem_post` if below `gateFloor²`. Bypassed when `hasActiveNotes` is true (worker has active/held notes that need note-offs). `hasActiveNotes` (atomic bool) and `activeNotesBits` (atomic uint64_t) added to `RangeStateBase`; updated by `applyNotesDiff` after every inference cycle.
21. **Staccato same-note re-hit detection** — Three changes: (a) `ONSET_BLANK_MS` 50→25 ms (also exposed as LV2 port 8 `onset_blank_ms`, configurable in tune via `onset_blank_ms` conf key / `--onset-blank` CLI); (b) `fireProv` in impl.cpp checks `activeNotesBits` and sends MIDI note-OFF before note-ON on same-note re-hit; (c) `processSynth` in tune.cpp releases existing voice for same pitch before starting new one. (The `applyNotesDiff` returning-note OFF+ON was also added here but later removed — see item 25.)
22. **Default mode changed to mono** — `plugin.ttl` mode default 0→1; `NoteRangeConfig::mode` default `POLY`→`MONO`; `neuralnote_tune.conf` updated to `mode = mono`.
23. **MIDI fan-out + CMake install targets** — `LV2/neuralnote-connect.sh` fans `ZynMidiRouter:ch0_out` out to extra synth chains. `LV2/neuralnote-connect.service` runs it at boot (After=zynthian.service). CMakeLists.txt `install()` targets deploy the bundle, script, and service; post-install code runs daemon-reload, conditionally enables the service, and restarts Zynthian. `LV2_INSTALL_PATH` cache variable (default `/zynthian/zynthian-plugins/lv2`). Zynthian `SINGLE_ACTIVE_CHANNEL` must be `0` for all chains to receive MIDI simultaneously.
24. **Cancel grace — legato retrigger fix** — When OBP+MPM fire a provisional, the snapshot dispatched at that moment has a ring buffer mostly filled with the previous note's audio (~3 ms of new note, ~77 ms of old). The CNN would cancel the correct provisional because it couldn't see enough frames of the new note. Fix: added `provLastSeenByCNN` and `provCancelGrace` (both worker-only ints) to `RangeStateBase`. On the first CNN cycle for each new provisional, if the CNN would cancel, suppress it by forcing the note into `newBits` so `applyNotesDiff` treats it as confirmed. The next cycle (with a full window) decides for real.
25. **Returning-note OFF+ON removed — sustained-note stutter fix** — `applyNotesDiff`'s returning-note path was sending OFF+ON whenever a held note reappeared in CNN output, including on sustained notes where CNN confidence fluctuated cycle-to-cycle. The note is never actually silent during hold (no OFF is sent when entering hold), so the OFF+ON was spurious. Fix: returning notes now just cancel the hold and continue playing. Same-note staccato re-hits are handled at the provisional level by `fireProv` reading `activeNotesBits`.
26. **Synth event logging (`>>`) in tune.cpp** — `processSynth` drain loop now prints every note-ON and note-OFF sent to the synth engine, prefixed with `>>`. Allows distinguishing CNN/worker-sourced events from provisional events and diagnosing stutters at the synth level.
27. **`LATENCY_OPTIMIZATIONS.md` removed** — Content superseded by `CLAUDE.md` and `LV2/README.md`.
28. **SwiftF0 `swiftmono` mode** — Third polyphony mode (mode=2). Replaces BasicPitch CNN with SwiftF0 (389 KB ONNX, 95K params, 16 kHz input). OBP+HPS+MPM provisional path unchanged. Worker detects `modeVal==2`, resamples 22050→16 kHz inline, calls `SwiftF0Detector::infer()` which returns a single MIDI note (median of confident frames) or -1. Cancel grace applies identically. `SwiftF0Detector.h` is header-only, uses ORT C++ API directly (`onnxruntime_cxx_api.h`). Tensor names hardcoded: `input_audio` / `pitch_hz` / `confidence`. Model lives at `swift_f0_model.onnx` in bundle root. `swift_threshold` (default 0.5 — calibrated for guitar; paper default 0.9 is for vocals) configurable via conf key / `--swift-threshold` CLI (tune only; LV2 uses hardcoded 0.5f). LV2 loads model at `instantiate()` time with graceful fallback if absent. `plugin.ttl` port 7 max 1→2 with `SwiftMono` scale point.
29. **SwiftF0 hold cycles + log dedup** — SwiftF0 cycles ~3–5× faster than BasicPitch, so `holdCycles=16` caused ~640ms of excess sustain after note-off. Added `swift_hold_cycles` per-range config (default 2); `applyNotesDiff` takes optional `holdCyclesOverride` param, used in swiftmono worker path. Also suppressed repeated SwiftF0 log lines in tune.cpp — only prints when detected note changes per range (`lastSwiftPrint` field on `RangeState`).
30. **Decay-tail ghost suppression** — SwiftF0 detecting harmonics in decaying string audio (e.g. D4 decay → D3 ghost in lower range). Fix: in swiftmono worker, if SwiftF0 wants a new note-ON (`newBits & ~activeNotes`) and last onset was >250 ms ago (`ONSET_GATE_S`), suppress `newBits`. Uses `totalSamples`/`lastOnsetSample` atomics (audio→worker). Every legitimate new note starts with an onset in mono mode.
31. **Onset grace (from-silence only) + cancel grace for swiftmono** — Two complementary grace mechanisms: (a) From-silence onset grace (`swiftOnsetGrace`=2, `swiftGraceStaleNote`): when `activeNotes==0` and onset fires, suppresses stale ring detections (e.g. old E2 in ring when B2 is played). Smart stale-note tracking: remembers stale note on cycle 1, allows different notes on cycle 2. Only active when no notes playing. (b) Cancel grace (`provCancelGrace`, item 24) re-enabled for swiftmono: protects provisional for 1 cycle until SwiftF0 confirms. Tied to provisional timing, not dispatch timing — more reliable for mid-note transitions. Velocity set to 127 so provisional wins mono reduction over stale SwiftF0 detections.
32. **±1 semitone onset suppression** — On onset-dispatched snapshots in swiftmono worker, if SwiftF0 detects a note ±1 semitone from the active note, suppress as transitional artifact (mixed old+new audio in ring). Eliminates D#4 between E4→D4 transitions. Larger changes pass through (handled by cancel grace or normal note-change logic).
33. **HPF pick detector (`PickDetector`)** — Replaces RMS as primary onset trigger. 1st-order IIR HPF at 3 kHz isolates broadband pick "snap"; dual-EMA envelope (fast 0.1 ms / slow 20 ms) measures attack speed. Two-tier threshold: ratio ≥ 10 = immediate onset (tier 1); ratio 4–10 = confirmed only if RMS also agrees (tier 2, logged with `+RMS` suffix). RMS onset remains as fallback for non-pick onsets (hammer-ons, volume swells). PICK fires provide sub-buffer sample precision for `lastOnsetSample`. `envSlow = envFast` on fire (snap) forces peak to rebuild from new transient. Internal blank period (25 ms) independent from RMS blank. Runs after noise gate (`!gated`).
34. **MPM-trust for pitch (OBP as onset gate)** — OBP detects harmonics instead of fundamentals for low notes (E2-B2 range) due to Butterworth lowpass cutoff being near the 2nd harmonic. Three-tier consensus: (a) immediate: OBP+MPM agree → fire with agreed pitch; (b) pending retry: OBP+MPM disagree → save pending, retry with more MPM data → trust MPM (more accurate at higher fill); (c) OBP expiry fallback: OBP window expired with no vote → try MPM alone. MPM corrects OBP's harmonic errors (e.g. A#2→G2, G#4→E4). Mono cross-range: suppress pending/fallback if any other range has `activeNotesBits!=0` or `provNote!=-1`. Bidirectional harmonic check: `abs(diff)==12||24` catches both harmonics and sub-harmonics.
35. **Provisional cooldown (200ms)** — After a provisional fires in a range, same-note re-triggers are blocked for 200ms (`provCooldownRemain`/`provCooldownNote` on `RangeStateBase`). Prevents RMS re-trigger stutters from the string's energy buildup. Only tier-1 PICK (ratio ≥ 10, genuine new attack) resets cooldown. Per-range, per-note: different notes are not blocked.
36. **Redundant provisional skip** — `fireProv` returns early if the same note is already in `activeNotesBits`. Prevents OFF+ON stutters when RMS re-triggers fire provisionals for an already-playing note.
37. **OBP improvements** — (a) Re-trigger onsets extend OBP window without resetting OBP/voting/MPM state (preserves accumulated readings). (b) `resetDetection()` preserves Butterworth filter state to avoid transient oscillation on pick attack; only resets Schmitt trigger and voting. (c) Lowpass multiplier 1.5×→1.2× midiHigh for tighter harmonic rejection.
38. **E2-B2 window 150→100ms, swift_hold_cycles 2→3** — Reduces SwiftF0 latency from ~200ms to ~60-90ms for low notes (dispatch interval 75→50ms). Hold increased to 150ms (3×50ms) to give SwiftF0 time to confirm before hold expires.
