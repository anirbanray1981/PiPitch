# NeuralNote Guitar2MIDI — LV2 Plugin

Real-time audio-to-MIDI conversion for LV2 hosts (Zynthian, Ardour, Carla, etc.).
Guitar (or any mono tonal audio) → two-phase pitch detection → MIDI note events.

---

## How it works

The plugin uses a **two-phase pipeline** to minimise perceived latency while
maintaining accuracy:

1. **Fast provisional note** (5–30 ms after pick attack)
   On every onset the audio thread runs:
   - **PickDetector**: 1st-order IIR HPF at 3 kHz isolates pick "snap";
     dual-EMA envelope (fast 0.1 ms / slow 20 ms) fires on transient ratio.
     Two-tier: ratio >= 10 = immediate onset; 4–10 = confirmed by RMS.
   - **OBP** (OneBitPitchDetector): 4th-order Butterworth LP → adaptive Schmitt
     trigger → period averaging.  Requires 4 consecutive agreeing readings
     (`N_CONSEC = 4`).
   - **HPS** (bit-parallel Harmonic Product Spectrum): cross-range OBP
     registers are shifted ×2 and ×3 to find the true fundamental.
   - **MPM** (McLeod Pitch Method, Pi 5 only): FFT autocorrelation + NSDF +
     parabolic interpolation for a second independent pitch estimate.

   A provisional MIDI note-ON fires as soon as OBP + HPS + MPM all agree on
   the same pitch.  On Pi 4 (MPM not compiled in), OBP + HPS agreement is
   sufficient.  Provisional notes above MIDI 76 (E5) are rejected — OBP is
   unreliable in that register.

2. **Inference confirmation** (worker thread)
   A background worker thread (`runWorkerCommon<Hooks>` in `NeuralNoteShared.h`)
   runs inference on a ring-buffer snapshot.  The inference engine depends on
   the polyphony mode:

   | Mode | Engine | Latency (Pi 5) | Description |
   |------|--------|---------------|-------------|
   | **poly** (0) | BasicPitch CNN | ~95 ms | Polyphonic; confirms/corrects/cancels provisional |
   | **mono** (1) | BasicPitch CNN | ~95 ms | Monophonic (single highest-velocity note) |
   | **swiftmono** (2) | SwiftF0 ONNX | ~10–20 ms | Monophonic; 389 KB model, 16 kHz input |
   | **swiftpoly** (3) | SwiftF0 + BasicPitch | ~100 ms | SwiftF0 for fast note-ON, BasicPitch for sustain/OFF |

```
JACK callback (RT thread)
  ├─ PickDetector  (HPF 3 kHz → dual-EMA → transient ratio)
  ├─ RMS onset     (fallback for hammer-ons / volume swells)
  ├─ OBP + HPS + MPM  ──────────────────────────── provisional fire
  ├─ Ring buffer fill  (22 050 Hz resampled audio)
  └─ Snapshot dispatch → worker thread (lockless SPSC)
       │
       ├─ [mode 0/1]  BasicPitch CNN (~95 ms)
       │     └─ buildNNBits �� cancel grace → applyNotesDiff
       │
       ├─ [mode 2]    SwiftF0 (~10–20 ms)
       │     └─ resample 22050→16 kHz → infer → onset grace
       │        → ghost suppression → note-change confirmation
       │        → cancel grace → applyNotesDiff
       │
       └─ [mode 3]    SwiftF0 + BasicPitch
             └─ SwiftF0 (~5 ms) → BasicPitch (~95 ms) → merge
                keep-alive bridge → active cancellation
                → cancel grace → applyNotesDiff
```

### Worker thread architecture

The worker loop is implemented once as a template function
`runWorkerCommon<Hooks>` in `NeuralNoteShared.h`.  Each consumer provides a
hooks struct:

| Consumer | Hooks struct | Logging |
|----------|-------------|---------|
| `neuralnote_impl.cpp` (LV2) | `ImplWorkerHooks` | No-op (zero overhead) |
| `neuralnote_tune.cpp` (JACK) | `TuneWorkerHooks` | printf diagnostics |

The hooks struct provides state accessors (13 methods for parameters, ranges,
SwiftF0, sample rate, etc.) and event callbacks (6 hooks: `onSwiftResult`,
`onSwiftPolyResult`, `onCNNOutcome`, `onNotesChanged`, `onMonoKill`,
`onShutdownOff`).

### Threading

- **Audio callback** (RT thread): onset detection, OBP, HPS, MPM, ring fill, dispatch.
- **Worker thread** (one, shared across all ranges): inference, `applyNotesDiff`, MIDI output.
- All comms are lockless: `SnapshotChannel` (SPSC atomic + semaphore), `MidiOutQueue` (SPSC ring).

---

## LV2 ports

| Index | Symbol | Type | Dir | Default | Range | Description |
|-------|--------|------|-----|---------|-------|-------------|
| 0 | `audio_in` | AudioPort | In | — | — | Mono audio in (guitar / instrument) |
| 1 | `midi_out` | AtomPort | Out | — | — | Transcribed MIDI note events |
| 2 | `audio_out` | AudioPort | Out | — | — | Silent audio through (for Zynthian chain classification) |
| 3 | `threshold` | ControlPort | In | 0.6 | 0.1–1.0 | Onset sensitivity — higher = fewer notes, fewer false positives |
| 4 | `gate_floor` | ControlPort | In | 0.003 | 0.0–0.1 | Noise gate floor (linear RMS); CNN dispatch skipped below this when no notes are active |
| 5 | `amp_floor` | ControlPort | In | 0.65 | 0.0–1.0 | BasicPitch amplitude floor; CNN frames below this confidence are discarded |
| 6 | `frame_threshold` | ControlPort | In | 0.5 | 0.05–0.95 | Per-frame CNN confidence threshold |
| 7 | `mode` | ControlPort | In | 1 | 0–3 | Polyphony: 0 = Poly · 1 = Mono · 2 = SwiftMono · 3 = SwiftPoly |
| 8 | `onset_blank_ms` | ControlPort | In | 25 | 10–100 ms | Re-trigger suppression window after each onset |

---

## Note range configuration

The plugin splits the MIDI range into independent **inference contexts**.
Each range runs its own ring buffer and CNN window tuned for that register.

**`neuralnote_ranges.conf`** (shipped inside the LV2 bundle — range sections only):

```ini
[range]
name              = E2-B2
midi_low          = 40        # lowest MIDI note (inclusive)
midi_high         = 47        # highest MIDI note (inclusive)
window            = 100       # CNN capture window in ms
min_note_length   = 6         # minimum CNN frames (1 frame ≈ 11.6 ms)
hold_cycles       = 4         # inference cycles before sending note-OFF
swift_hold_cycles = 3         # hold cycles for SwiftF0 (faster cycle time)

[range]
name            = C3-B3
midi_low        = 48
midi_high       = 59
window          = 100
min_note_length = 4
hold_cycles     = 16

[range]
name            = C4-B4
midi_low        = 60
midi_high       = 71
window          = 80
min_note_length = 3
hold_cycles     = 16

[range]
name            = C5-G#5
midi_low        = 72
midi_high       = 80
window          = 60
min_note_length = 2
hold_cycles     = 16

[range]
name            = A5+
midi_low        = 81
midi_high       = 127
window          = 80
min_note_length = 3
hold_cycles     = 16
```

Notes outside all defined ranges are silently discarded.

Global parameters (`gate_floor`, `amp_floor`, `threshold`, `frame_threshold`,
`mode`, `onset_blank_ms`) are **not** in the bundle config — their values come
from the LV2 port defaults in `plugin.ttl` and can be changed by the host at
runtime.

---

## Prerequisites

| Tool | Minimum version |
|------|----------------|
| CMake | 3.16 |
| C++17 compiler (GCC / Clang) | GCC 10 / Clang 12 |
| LV2 development headers | 1.18 |
| pkg-config | any |
| JACK development headers | 1.9 (`neuralnote_tune` only) |
| FFTW3 single-precision | any (`neuralnote_tune` only) |

```bash
# Debian / Raspberry Pi OS
sudo apt-get install -y \
    build-essential cmake pkg-config \
    lv2-dev libjack-jackd2-dev libfftw3-dev
```

### ONNX Runtime — Linux aarch64

The bundled `ThirdParty/onnxruntime/lib/libonnxruntime.a` is macOS-only.
A pre-built Linux aarch64 shared library is staged at:

```
ThirdParty/onnxruntime/lib-linux-aarch64/libonnxruntime.so
```

Sourced from the official `onnxruntime` pip package (v1.24.4, generic aarch64).
To re-fetch:

```bash
pip3 install onnxruntime --break-system-packages
ORT_CAPI=$(python3 -c \
    "import onnxruntime,os; print(os.path.dirname(onnxruntime.__file__))")/capi
mkdir -p ThirdParty/onnxruntime/lib-linux-aarch64
cp  "$ORT_CAPI"/libonnxruntime.so.*.* \
    ThirdParty/onnxruntime/lib-linux-aarch64/libonnxruntime.so
ln -sf libonnxruntime.so \
    ThirdParty/onnxruntime/lib-linux-aarch64/libonnxruntime.so.1
```

> **Note:** The Debian `libonnxruntime-dev` package is compiled for ARMv8.2-A
> and causes SIGILL on Raspberry Pi 4 (Cortex-A72 / ARMv8-A).  Use the pip
> package instead.

---

## Building

```bash
# From the repository root
cmake -B build_lv2 \
      -DBUILD_LV2=ON \
      -DCMAKE_BUILD_TYPE=Release

cmake --build build_lv2 -j$(nproc)
```

### Build targets

| Target | Output | Platform |
|--------|--------|----------|
| `NeuralNoteGuitar2Midi_LV2` | `neuralnote_guitar2midi.so` | All — LV2 wrapper; selects impl at runtime via `AT_HWCAP` |
| `NeuralNoteImpl_NEON` | `neuralnote_impl_neon.so` | Pi 4 (ARMv8-A, NEON) — no MPM |
| `NeuralNoteImpl_ARMv82` | `neuralnote_impl_armv82.so` | Pi 5 (ARMv8.2-A, dotprod+fp16) — MPM enabled |
| `neuralnote_tune` | `neuralnote_tune` | JACK tuning tool (requires JACK + FFTW3f) |
| `latency_bench` | `latency_bench` | Offline latency benchmark |

On success the bundle is assembled at:

```
build_lv2/neuralnote_guitar2midi.lv2/
├── manifest.ttl
├── plugin.ttl
├── neuralnote_guitar2midi.so       ← LV2 host loads this
├── neuralnote_impl_neon.so         ← Pi 4 impl (selected at runtime)
├── neuralnote_impl_armv82.so       ← Pi 5 impl (selected at runtime)
├── neuralnote_ranges.conf          ← per-range tuning
├── swift_f0_model.onnx             ← SwiftF0 model (swiftmono/swiftpoly)
└── ModelData/
    ├── cnn_contour_model.json
    ├── cnn_note_model.json
    ├── cnn_onset_1_model.json
    ├── cnn_onset_2_model.json
    └── features_model.ort
```

### Pi 5 — manual rebuild (no cmake re-run)

cmake cannot be re-run on the Pi because the JUCE submodule is absent.
Recompile individual targets using the extracted flags:

```bash
cd /root/neuralnote_build

DEFINES="-DNEURALNOTE_IMPL_NAME=\"neuralnote_impl_armv82\" \
  -DRTNEURAL_DEFAULT_ALIGNMENT=16 -DRTNEURAL_NAMESPACE=RTNeural \
  -DRTNEURAL_USE_EIGEN=1 -DSAVE_DOWNSAMPLED_AUDIO=0 \
  -DUSE_TEST_NOTE_FRAME_TO_TIME=0 -Dneuralnote_impl_armv82_EXPORTS"
INCLUDES="-I/root/neuralnote_src/LV2 -I/root/neuralnote_src/Lib/Model \
  -I/root/neuralnote_src/Lib/Utils \
  -I/root/neuralnote_src/ThirdParty/RTNeural \
  -I/root/neuralnote_src/ThirdParty/onnxruntime/include \
  -I/root/neuralnote_src/ThirdParty/RTNeural/RTNeural/../modules/json \
  -I/root/neuralnote_src/ThirdParty/RTNeural/RTNeural/.. \
  -I/root/neuralnote_src/ThirdParty/RTNeural/modules/Eigen"

# Rebuild impl
/usr/bin/c++ $DEFINES $INCLUDES \
  -mcpu=cortex-a76 -mtune=cortex-a76 -O3 -DNDEBUG -fPIC -O2 \
  -march=armv8.2-a+dotprod+fp16 -std=gnu++17 \
  -c /root/neuralnote_src/LV2/neuralnote_impl.cpp \
  -o CMakeFiles/neuralnote_impl_armv82.dir/LV2/neuralnote_impl.cpp.o
bash CMakeFiles/neuralnote_impl_armv82.dir/link.txt

# Rebuild tune tool
DEFINES_TUNE="-DRTNEURAL_DEFAULT_ALIGNMENT=16 -DRTNEURAL_NAMESPACE=RTNeural \
  -DRTNEURAL_USE_EIGEN=1 -DSAVE_DOWNSAMPLED_AUDIO=0 -DUSE_TEST_NOTE_FRAME_TO_TIME=0"
/usr/bin/c++ $DEFINES_TUNE $INCLUDES \
  -mcpu=cortex-a76 -mtune=cortex-a76 -O3 -DNDEBUG -fPIE -O2 -std=gnu++17 \
  -c /root/neuralnote_src/LV2/neuralnote_tune.cpp \
  -o CMakeFiles/neuralnote_tune.dir/LV2/neuralnote_tune.cpp.o
bash CMakeFiles/neuralnote_tune.dir/link.txt
```

---

## CMake options

| Option | Default | Description |
|--------|---------|-------------|
| `BUILD_LV2` | `OFF` | Enable all LV2 targets |
| `BUILD_LV2_TUNE` | `ON` | Build `neuralnote_tune` JACK tuning tool |
| `BUILD_LV2_BENCH` | `ON` | Build `latency_bench` offline benchmark |
| `LV2_WINDOW_MS` | `500` | Fallback CNN capture window in ms (overridden by range config) |
| `LV2_INSTALL_PATH` | `/zynthian/zynthian-plugins/lv2` | Bundle install destination |
| `LTO` | `ON` | Link-Time Optimisation |
| `RTNeural_Release` | `OFF` | Force Release optimisation for RTNeural in Debug builds |

---

## `neuralnote_tune` — JACK tuning tool

A standalone JACK application that mirrors the LV2 plugin logic with detailed
console logging and a built-in synth engine for audio feedback.  Used for live
parameter tuning on the target hardware.

```
neuralnote_tune [--bundle PATH] [--config PATH]
                [--threshold 0.6] [--frame-threshold 0.5]
                [--mode poly|mono|swiftmono|swiftpoly]
                [--swift-threshold 0.5]
                [--gate 0.003] [--amp-floor 0.65]
                [--onset-blank MS] [--window MS] [--hold-cycles N]
                [--waveform sine|saw|square]
                [--attack MS] [--release MS] [--volume 0.3]
```

CLI values override the config file.  Run from the build directory:

```bash
cd /root/neuralnote_build
./neuralnote_tune --config neuralnote_tune.conf --volume 1
```

`libonnxruntime.so.1` resolves via rpath — no `LD_LIBRARY_PATH` needed.
`libfftw3f` is statically linked.

### Console output

```
Time          Event  Note    MIDI#  Vel  Info
------------  -----  ------  -----  ---  ------------------------
[+1.876s]  --   MPM not ready (fill 256/2048)  OBP→F4 (65)  pending  [range C4-B4]
[+2.000s]  ON   A4   ( 69)  vel 101  [CNN  win 80ms  inf 10ms  range C4-B4]
[+2.000s]  >>   A4   ( 69)  vel 101  [synth ON   range C4-B4]
[+2.516s]  ON   B4   ( 71)  vel 100  [1-bit provisional  range C4-B4]
[+2.658s]  ON   B4   ( 71)  vel 104  [CNN  win 80ms  inf 10ms  range C4-B4]
[+2.658s]  >>   B4   ( 71)  vel 104  [synth ON   range C4-B4]
[+3.271s]  OFF  A4   ( 69)
[+3.271s]  >>   A4   ( 69)            [synth OFF  range C4-B4]
```

| Marker | Meaning |
|--------|---------|
| `ON` / `OFF` | MIDI note event (provisional or CNN-confirmed) |
| `>>` | Event actually sent to the synth engine this callback |
| `--` | Diagnostic only — OBP/MPM/CNN state, no MIDI sent |

### `neuralnote_tune.conf`

Full config including global keys (the LV2 bundle's `neuralnote_ranges.conf`
contains range sections only):

```ini
gate_floor       = 0.003
amp_floor        = 0.65
threshold        = 0.6
frame_threshold  = 0.5
mode             = mono
onset_blank_ms   = 25
swift_threshold  = 0.5

[range]
name            = E2-B2
midi_low        = 40
midi_high       = 47
window          = 150
min_note_length = 6
hold_cycles     = 4
...
```

---

## Deploying to Zynthian

### Paths

| Location | Path |
|----------|------|
| Pi 5 source mirror | `/root/neuralnote_src/` |
| Pi 5 build directory | `/root/neuralnote_build/` |
| LV2 bundle (live) | `/zynthian/zynthian-plugins/lv2/neuralnote_guitar2midi.lv2/` |

### Manual deploy

```bash
# Always copy TTL + conf + model from source before deploying
cp /root/neuralnote_src/LV2/plugin.ttl              /root/neuralnote_build/neuralnote_guitar2midi.lv2/
cp /root/neuralnote_src/LV2/neuralnote_ranges.conf  /root/neuralnote_build/neuralnote_guitar2midi.lv2/
cp /root/neuralnote_src/LV2/neuralnote_tune.conf    /root/neuralnote_build/neuralnote_guitar2midi.lv2/
cp /root/neuralnote_src/LV2/swift_f0_model.onnx     /root/neuralnote_build/neuralnote_guitar2midi.lv2/
cp -r /root/neuralnote_build/neuralnote_guitar2midi.lv2/* \
    /zynthian/zynthian-plugins/lv2/neuralnote_guitar2midi.lv2/
systemctl restart zynthian
```

> cmake does not rebuild TTL, conf, or ONNX model files — always copy them manually.

### `make install`

```bash
cd /root/neuralnote_build
cmake --install .
```

Deploys the bundle to `LV2_INSTALL_PATH`, installs `neuralnote_tune` and
`neuralnote-connect.sh` to `/usr/local/bin/`, installs and conditionally
enables `neuralnote-connect.service`, then restarts Zynthian and the service.

---

## MIDI fan-out — multiple synth chains

Zynthian's autoconnect only routes NeuralNote MIDI to the chain it manages.
Two files drive additional chains in parallel:

**`neuralnote-connect.sh`** polls for each JACK port then calls `jack_connect`
to wire `ZynMidiRouter:ch0_out` to every extra destination listed in
`EXTRA_DSTS`.  Edit the array to match your synth chain layout:

```bash
EXTRA_DSTS=(
    "fluidsynth:midi_00"
    "LinuxSampler:midi_in_0"
)
```

**`neuralnote-connect.service`** is a systemd oneshot unit that runs the script
after `zynthian.service` starts at boot:

```ini
[Unit]
After=zynthian.service
Requires=zynthian.service

[Service]
Type=oneshot
ExecStart=/usr/local/bin/neuralnote-connect.sh
```

Manual service management:

```bash
systemctl enable  neuralnote-connect.service   # persists across reboots
systemctl restart neuralnote-connect.service   # re-run immediately
journalctl -u neuralnote-connect.service        # view output
```

### Per-chain MIDI setup

For all chains to receive NeuralNote MIDI simultaneously:

1. Disable **Single Active Channel** so Zynthian routes MIDI to all chains,
   not just the focused one:
   ```bash
   # /zynthian/config/midi-profiles/default.sh
   export ZYNTHIAN_MIDI_SINGLE_ACTIVE_CHANNEL="0"
   ```
   Then restart Zynthian.

2. Set each synth chain's **MIDI channel to 1** (0-indexed channel 0) in the
   Zynthian UI to match what `ZynMidiRouter:ch0_out` carries.

---

## Key source files

| File | Purpose |
|------|---------|
| `neuralnote_guitar2midi.cpp` | LV2 entry point; `dlopen`-selects impl via `AT_HWCAP` at runtime |
| `neuralnote_impl.cpp` | LV2 plugin — RT callback, `ImplWorkerHooks`, MIDI output |
| `neuralnote_tune.cpp` | JACK tuning tool — `TuneWorkerHooks` with logging and synth engine |
| `NeuralNoteShared.h` | Shared code: constants, structs, pipeline helpers, `buildNNBits`, `applyNotesDiff`, `runWorkerCommon<Hooks>` |
| `SwiftF0Detector.h` | SwiftF0 ONNX wrapper: `infer()` → median-confident MIDI note or -1 |
| `swift_f0_model.onnx` | SwiftF0 model (389 KB, 95K params, 16 kHz, 16 ms/frame) |
| `MidiNotes.h` | Named MIDI note constants (C2–E6) |
| `NoteRangeConfig.h` | `NoteRange` / `RangeConfig` / `PlayMode` structs and INI config parser |
| `OneBitPitchDetector.h` | OBP: Butterworth LP → Schmitt trigger → period averaging |
| `McLeodPitchDetector.h` | MPM: FFT autocorrelation → NSDF → parabolic interpolation |
| `plugin.ttl` | LV2 port definitions, defaults, scale points |
| `manifest.ttl` | LV2 bundle manifest |
| `neuralnote_ranges.conf` | Default range config shipped in the LV2 bundle |
| `neuralnote_tune.conf` | Full config for the tuning tool (includes global keys) |
| `neuralnote-connect.sh` | JACK MIDI fan-out script |
| `neuralnote-connect.service` | systemd unit for the fan-out script |

---

## Key constants

| Constant | Value | Meaning |
|----------|-------|---------|
| `NOTE_BASE` | 40 | Lowest MIDI note in bitmap (E2) |
| `NOTE_COUNT` | 49 | Bitmap covers MIDI 40–88 (E2–E6) |
| `PLUGIN_SR` | 22 050 Hz | Resampled rate fed to BasicPitch CNN |
| `OBP_NOTE_CAP` | 76 | OBP provisionals above E5 are rejected |
| `N_CONSEC` | 4 | Consecutive agreeing OBP readings before provisional fires |
| `ONSET_RATIO` | 3.0× | RMS fallback onset threshold (block RMS vs. smoothed background) |
| `ONSET_BLANK_MS` | 25 ms | Default re-trigger suppression (overridden by port 8 / conf key) |
| `ONSET_GATE_S` | 0.25 s | Decay-tail ghost suppression: max time after onset for SwiftF0 new notes |
| `PICK_HIGH_TIER` | 10.0 | PickDetector fast/slow ratio for immediate onset (tier 1) |
| `MIN_FRESH_FLOOR` | 25 ms | Minimum fresh audio before CNN dispatch |
| `MPM_FFTSIZE` | 4 096 | Zero-padded FFT size for MPM autocorrelation |
| `MPM_K` | 0.86 | NSDF key-maximum threshold |
| `SWIFT_POLY_KEEPALIVE` | 2 | Cycles to keep SwiftF0 note alive awaiting BasicPitch confirmation |
