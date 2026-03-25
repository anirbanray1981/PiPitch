# NeuralNote Guitar2MIDI — LV2 Plugin

Audio-to-MIDI transcription for LV2 hosts (Ardour, Carla, Zynthian, Pisound, etc.).
Guitar (or any mono tonal audio) → NeuralNote's BasicPitch engine → MIDI note events.

## Ports

| Index | Symbol | Type | Dir | Default | Description |
|-------|--------|------|-----|---------|-------------|
| 0 | `audio_in` | AudioPort | In | — | Mono audio in (guitar / instrument) |
| 1 | `midi_out` | AtomPort (MIDI Sequence) | Out | — | Transcribed MIDI note events |
| 2 | `threshold` | ControlPort | In | 0.5 | Note sensitivity 0.1–1.0 (lower = fewer notes, less harmonics) |
| 3 | `mode` | ControlPort | In | 1 | Latency mode: 0 = Fast 300 ms · 1 = Medium 500 ms · 2 = Slow 1000 ms |
| 4 | `audio_out` | AudioPort | Out | — | Silent audio through (enables audio-effect classification in Zynthian) |
| 5 | `gate_floor` | ControlPort | In | 0.003 | Noise gate floor as linear RMS (≈ −50 dBFS); 0 = disabled |
| 6 | `min_dur` | ControlPort | In | 100 ms | Minimum note duration in ms; filters brief harmonic artefacts |
| 7 | `amp_floor` | ControlPort | In | 0.65 | Minimum note amplitude 0.0–1.0; filters weak model artefacts |

---

## Prerequisites

### All platforms

| Tool | Minimum version |
|------|----------------|
| CMake | 3.16 |
| C++17 compiler (GCC / Clang) | GCC 10 / Clang 12 |
| LV2 development headers | 1.18 |
| pkg-config | any |

### Linux (including Raspberry Pi 4 / Pi 5 / Pisound)

```bash
# Debian / Raspberry Pi OS
sudo apt-get update
sudo apt-get install -y \
    build-essential cmake pkg-config \
    lv2-dev liblilv-dev lilv-utils
```

#### ONNX Runtime — Linux aarch64 (Raspberry Pi)

The bundled `ThirdParty/onnxruntime/lib/libonnxruntime.a` is a **macOS-only** universal binary.
A pre-built Linux aarch64 shared library is included in:

```
ThirdParty/onnxruntime/lib-linux-aarch64/libonnxruntime.so
```

This was sourced from the official `onnxruntime` pip package (v1.24.4, generic aarch64 build).
If you want to re-fetch it yourself:

```bash
pip3 install onnxruntime --break-system-packages
ORT_CAPI=$(python3 -c "import onnxruntime,os; print(os.path.dirname(onnxruntime.__file__))")/capi
mkdir -p ThirdParty/onnxruntime/lib-linux-aarch64
cp  "$ORT_CAPI"/libonnxruntime.so.*.* \
    ThirdParty/onnxruntime/lib-linux-aarch64/libonnxruntime.so
ln -sf libonnxruntime.so \
    ThirdParty/onnxruntime/lib-linux-aarch64/libonnxruntime.so.1
```

CMake will find the library automatically in that staging directory.

> **Note:** The Debian package `libonnxruntime-dev` (v1.21+) as of 2025 is compiled for
> ARMv8.2-A and **causes SIGILL on Raspberry Pi 4** (Cortex-A72 / ARMv8-A).
> Use the pip package instead.

---

## Building

```bash
# From the repository root:
cmake -B build_lv2 \
      -DBUILD_LV2=ON \
      -DCMAKE_BUILD_TYPE=Release

cmake --build build_lv2 --target NeuralNoteGuitar2Midi_LV2 -j$(nproc)
```

### SIMD auto-detection

CMake probes the build host at configure time by **compiling and executing** a
small test program for each SIMD level.  Because the build runs directly on the
target Pi (no cross-compilation), a Pi 4 will naturally fail the ARMv8.2-A
`sdot` probe (SIGILL) and fall back to NEON, while a Pi 5 will pass it.

| Platform | Detected level | `-march` flag |
|----------|---------------|---------------|
| Raspberry Pi 5 (Cortex-A76) | `armv8.2-a+dotprod+fp16` | `-march=armv8.2-a+dotprod+fp16` |
| Raspberry Pi 4 (Cortex-A72) | `armv8-a+neon` | `-march=armv8-a+simd` |
| x86-64 with AVX2 | `avx2+fma` | `-mavx2 -mfma` |
| x86-64 without AVX2 | `sse4.2` | `-msse4.2` |
| Other | `baseline` | *(none)* |

The configure output will show the detected level:

```
-- LV2 SIMD: armv8.2-a+dotprod+fp16  (march=armv8.2-a+dotprod+fp16)
```

The detected level is also baked into the plugin binary as the
`NEURALNOTE_SIMD_LEVEL` string and logged by the LV2 host on load:

```
NeuralNote Guitar2MIDI: instantiated at 48000 Hz  [SIMD: armv8.2-a+dotprod+fp16]
```

To force a re-probe (e.g. after moving the build directory to a different
machine), wipe the cached results and re-run cmake:

```bash
cmake -B build_lv2 -UNEURALNOTE_HAVE_ARMV82_DOTPROD -UNEURALNOTE_HAVE_NEON \
      -UNEURALNOTE_HAVE_AVX2 -UNEURALNOTE_HAVE_SSE42 \
      -DBUILD_LV2=ON -DCMAKE_BUILD_TYPE=Release
```

On success the bundle is assembled at:

```
build_lv2/neuralnote_guitar2midi.lv2/
├── manifest.ttl
├── plugin.ttl
├── neuralnote_guitar2midi.so
└── ModelData/
    ├── cnn_contour_model.json
    ├── cnn_note_model.json
    ├── cnn_onset_1_model.json
    ├── cnn_onset_2_model.json
    └── features_model.ort
```

---

## Testing the build

### 1. Validate the bundle metadata

```bash
LV2_PATH=build_lv2 lv2info "https://github.com/DamRsn/NeuralNote/guitar2midi"
```

Expected output (abbreviated):

```
https://github.com/DamRsn/NeuralNote/guitar2midi
    Name:    NeuralNote Guitar2MIDI
    Port 0:  audio_in   (AudioPort, Input)
    Port 1:  midi_out   (AtomPort, Output)
    Port 2:  threshold  (ControlPort, Input, default=0.5)
    Port 3:  mode       (ControlPort, Input, default=1)
    Port 4:  audio_out  (AudioPort, Output)
    Port 5:  gate_floor (ControlPort, Input, default=0.003)
    Port 6:  min_dur    (ControlPort, Input, default=100)
    Port 7:  amp_floor  (ControlPort, Input, default=0.65)
```

### 2. Verify the shared library loads and exports `lv2_descriptor`

```bash
LD_LIBRARY_PATH=$(pwd)/ThirdParty/onnxruntime/lib-linux-aarch64 \
python3 - <<'EOF'
import ctypes
lib = ctypes.CDLL("build_lv2/neuralnote_guitar2midi.lv2/neuralnote_guitar2midi.so")
lib.lv2_descriptor.restype  = ctypes.c_void_p
lib.lv2_descriptor.argtypes = [ctypes.c_uint32]
assert lib.lv2_descriptor(0) != 0, "descriptor(0) must be non-null"
assert lib.lv2_descriptor(1) is None, "descriptor(1) must be null"
print("PASS — lv2_descriptor entry point is correct")
EOF
```

### 3. Scan with lv2lint (optional, requires `lv2lint` package)

```bash
sudo apt-get install -y lv2lint
LD_LIBRARY_PATH=$(pwd)/ThirdParty/onnxruntime/lib-linux-aarch64 \
LV2_PATH=build_lv2 \
lv2lint "https://github.com/DamRsn/NeuralNote/guitar2midi"
```

---

## Installing

Copy the bundle to an LV2 search path that your host scans:

```bash
# User install (recommended)
cp -r build_lv2/neuralnote_guitar2midi.lv2 ~/.lv2/

# System-wide install
sudo cp -r build_lv2/neuralnote_guitar2midi.lv2 /usr/local/lib/lv2/
```

The plugin needs the onnxruntime shared library at runtime.
Either install it system-wide:

```bash
sudo cp ThirdParty/onnxruntime/lib-linux-aarch64/libonnxruntime.so   /usr/local/lib/
sudo cp ThirdParty/onnxruntime/lib-linux-aarch64/libonnxruntime.so.1 /usr/local/lib/
sudo ldconfig
```

Or set `LD_LIBRARY_PATH` before launching your LV2 host:

```bash
export LD_LIBRARY_PATH=/path/to/NeuralNote/ThirdParty/onnxruntime/lib-linux-aarch64:$LD_LIBRARY_PATH
ardour7  # or carla, jalv, etc.
```

---

## Quick functional test with jalv

```bash
sudo apt-get install -y jalv

# Install the bundle and onnxruntime first (see above), then:
jalv.gtk "https://github.com/DamRsn/NeuralNote/guitar2midi"
```

Connect a guitar/microphone audio source to Port 0 and route Port 1 (MIDI) to
a synthesizer or MIDI recorder. Play and watch notes appear.

---

## Architecture overview

The plugin uses a **ring buffer + background worker thread** design to keep `lv2_run()` non-blocking:

```
lv2_run()  (audio thread)
  ├─ drain pending MIDI events → Atom sequence output
  ├─ update control ports (threshold, mode, gate_floor, min_dur, amp_floor)
  ├─ noise gate: compute block RMS; if below gate_floor push zeros into ring
  ├─ resample audio block to 22050 Hz → circular ring buffer
  └─ when ring full: try_to_lock → hand snapshot to worker thread

Worker thread  (background)
  ├─ BasicPitch::setParameters(threshold, 0.5, min_dur)
  ├─ BasicPitch::transcribeToMIDI(snapshot)
  │     ├─ Features (ONNX CQT)  → stacked CQT frames
  │     └─ BasicPitchCNN (RTNeural) → note/onset/contour posteriorgrams
  ├─ filter events below amp_floor
  ├─ diff result against active note set → note-on / note-off events
  └─ append to pendingMidi (picked up by next lv2_run())
```

The ring buffer size is controlled by the `mode` port:

| Mode | Window | Typical latency |
|------|--------|----------------|
| 0 — Fast | 300 ms | ~380 ms (Pi 4) / ~200 ms (Pi 5) |
| 1 — Medium (default) | 500 ms | ~580 ms / ~380 ms |
| 2 — Slow | 1000 ms | ~1080 ms / ~880 ms |

---

## Key source files

| File | Purpose |
|------|---------|
| `LV2/neuralnote_guitar2midi.cpp` | LV2 plugin entry point (`lv2_descriptor`) and `dlopen` dispatch |
| `LV2/neuralnote_impl.cpp` | Plugin implementation (ports, ring buffer, worker thread, MIDI output) |
| `LV2/neuralnote_monitor.cpp` | Standalone JACK terminal monitor (see below) |
| `LV2/plugin.ttl` | LV2 metadata — ports, defaults, scale points |
| `LV2/BinaryData.h` | File-loading substitute for JUCE BinaryData |
| `LV2/NoteUtils.h` | JUCE-free stub for `Lib/Utils/NoteUtils.h` |
| `Lib/Model/BasicPitch.{h,cpp}` | Top-level transcription pipeline |
| `Lib/Model/Features.{h,cpp}` | ONNX CQT feature extraction |
| `Lib/Model/BasicPitchCNN.{h,cpp}` | RTNeural CNN inference |
| `Lib/Model/Notes.{h,cpp}` | Posteriorgram → note event conversion |

---

## Terminal notes monitor

`neuralnote_monitor` is a standalone JACK application that captures audio from
`system:capture_1`, runs BasicPitch inference in a background thread, and prints
detected notes to the terminal. Useful for testing without a full LV2 host.

Build alongside the plugin (enabled by default when `BUILD_LV2=ON`):

```bash
cmake -B build_lv2 -DBUILD_LV2=ON -DBUILD_LV2_MONITOR=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build_lv2 --target neuralnote_monitor -j$(nproc)
```

Usage:

```
neuralnote_monitor [--bundle PATH] [--threshold 0.1-1.0] [--gate 0.0-0.1]
                   [--min-dur MS] [--amp-floor 0.0-1.0] [--mode 0|1|2]

  --bundle PATH      Path to the .lv2 bundle dir containing ModelData/
  --threshold        Note sensitivity (default 0.5)
  --gate             Noise gate floor linear RMS (default 0.003 ≈ −50 dBFS)
  --min-dur          Minimum note duration ms (default 100)
  --amp-floor        Minimum note amplitude 0–1 (default 0.65)
  --mode             0=Fast/300ms  1=Medium/500ms  2=Slow/1000ms  (default 1)
```

Example output:

```
Time          Event       Note    MIDI#  Vel
------------  ----------  ------  -----  ---
[+0.512s]  NOTE ON   E4   ( 64)  vel  91
[+1.847s]  NOTE OFF  E4   ( 64)
```

The monitor uses the same defaults as the LV2 plugin for all parameters.

---

## Zynthian integration

The plugin appears in Zynthian's chain UI under **Audio Effect → Other →
NeuralNote Guitar2MIDI**. The silent `audio_out` port (index 4) is present
solely so Zynthian's `get_plugin_type()` classifies it as an audio effect.

The `midi_out` JACK port (`NeuralNote_Guitar2MIDI-01:midi_out`) carries the
transcribed notes. Route it to `ZynMidiRouter:dev*_in` and any synth chain
listening on the corresponding MIDI channel will receive the notes.

---

## CMake options

| Option | Default | Description |
|--------|---------|-------------|
| `BUILD_LV2` | `OFF` | Enable the LV2 plugin target |
| `BUILD_LV2_MONITOR` | `ON` | Build the terminal notes monitor (requires JACK; only when `BUILD_LV2=ON`) |
| `BUILD_LV2_BENCH` | `ON` | Build the latency benchmark (only when `BUILD_LV2=ON`) |
| `BUILD_UNIT_TESTS` | `OFF` | Build unit tests |
| `RTNeural_Release` | `OFF` | Force Release optimisation for RTNeural in Debug builds |
| `LTO` | `ON` | Link-Time Optimisation |
