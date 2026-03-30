# PiPitch — Real-Time Guitar-to-MIDI on Raspberry Pi

Real-time audio-to-MIDI conversion for LV2 hosts (Zynthian, Ardour, Carla, etc.).
Guitar (or any mono tonal audio) → two-phase pitch detection → MIDI note events.

Built on [Spotify's BasicPitch](https://github.com/spotify/basic-pitch) CNN and
[SwiftF0](https://arxiv.org/html/2508.18440v1) monophonic pitch estimator,
running on Raspberry Pi 5 via [RTNeural](https://github.com/jatinchowdhury18/RTNeural)
and [ONNX Runtime](https://github.com/microsoft/onnxruntime).

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
     trigger → period averaging.  Requires 4 consecutive agreeing readings.
   - **HPS** (bit-parallel Harmonic Product Spectrum): cross-range OBP
     registers are shifted ×2 and ×3 to find the true fundamental.
   - **MPM** (McLeod Pitch Method, Pi 5 only): FFT autocorrelation + NSDF +
     parabolic interpolation for a second independent pitch estimate.

   A provisional MIDI note-ON fires as soon as OBP + HPS + MPM all agree.
   Controllable via the `provisional` parameter (on / swift / none).

2. **Inference confirmation** (worker thread)
   A background worker thread (`runWorkerCommon<Hooks>` in `PiPitchShared.h`)
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
       │     └─ buildNNBits → cancel grace → applyNotesDiff
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
`runWorkerCommon<Hooks>` in `PiPitchShared.h`.  Each consumer provides a
hooks struct:

| Consumer | Hooks struct | Logging |
|----------|-------------|---------|
| `pipitch_impl.cpp` (LV2) | `ImplWorkerHooks` | No-op (zero overhead) |
| `pipitch_tune.cpp` (JACK) | `TuneWorkerHooks` | printf diagnostics |

### Threading

- **Audio callback** (RT thread): onset detection, OBP, HPS, MPM, ring fill, dispatch.
- **Worker thread** (one, shared across all ranges): inference, `applyNotesDiff`, MIDI output.
- All comms are lockless: `SnapshotChannel` (SPSC atomic + semaphore), `MidiOutQueue` (SPSC ring).

---

## Repository structure

```
PiPitch/
├── NeuralNote/              ← git submodule (BasicPitch model code + dependencies)
│   ├── Lib/Model/           ← BasicPitch CNN inference
│   ├── ThirdParty/RTNeural/ ← Neural network runtime
│   └── ThirdParty/onnxruntime/
├── LV2/                     ← PiPitch plugin code
│   ├── pipitch_impl.cpp     ← LV2 plugin (RT callback + ImplWorkerHooks)
│   ├── pipitch_tune.cpp     ← JACK tuning tool (TuneWorkerHooks + synth)
│   ├── pipitch.cpp           ← LV2 wrapper (CPU dispatch via dlopen)
│   ├── PiPitchShared.h      ← Shared: constants, pipeline, runWorkerCommon<Hooks>
│   ├── SwiftF0Detector.h    ← SwiftF0 ONNX wrapper
│   ├── plugin.ttl / manifest.ttl
│   ├── pipitch_ranges.conf  ← Per-range config (shipped in bundle)
│   ├── pipitch_tune.conf    ← Tune tool config (includes global keys)
│   ├── pipitch-connect.sh   ← JACK MIDI fan-out script
│   └── pipitch-connect.service
├── CMakeLists.txt           ← LV2 build (references NeuralNote/ submodule)
└── README.md
```

---

## LV2 ports

| Index | Symbol | Default | Range | Description |
|-------|--------|---------|-------|-------------|
| 0 | `audio_in` | — | — | Mono audio in |
| 1 | `midi_out` | — | — | MIDI output (atom sequence) |
| 2 | `audio_out` | — | — | Audio through |
| 3 | `threshold` | 0.6 | 0.1–1.0 | Onset sensitivity |
| 4 | `gate_floor` | 0.003 | 0.0–0.1 | Noise gate floor |
| 5 | `amp_floor` | 0.65 | 0.0–1.0 | BasicPitch amplitude floor |
| 6 | `frame_threshold` | 0.5 | 0.05–0.95 | Per-frame CNN confidence |
| 7 | `mode` | 1 | 0–3 | Poly / Mono / SwiftMono / SwiftPoly |
| 8 | `onset_blank_ms` | 25 | 10–100 | Re-trigger suppression (ms) |
| 9 | `provisional` | 0 | 0–2 | On / Swift / None — controls OBP+MPM pipeline |

---

## Building

```bash
# From the repository root (requires NeuralNote submodule)
git clone --recurse-submodules https://github.com/anirbanray1981/PiPitch.git
cd PiPitch

cmake -B build -DBUILD_LV2=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
```

### Build targets

| Target | Output | Platform |
|--------|--------|----------|
| `PiPitch_LV2` | `pipitch.so` | LV2 wrapper; selects impl at runtime via `AT_HWCAP` |
| `PiPitchImpl_NEON` | `pipitch_impl_neon.so` | Pi 4 (ARMv8-A, NEON) — no MPM |
| `PiPitchImpl_ARMv82` | `pipitch_impl_armv82.so` | Pi 5 (ARMv8.2-A, dotprod+fp16) — MPM enabled |
| `pipitch_tune` | `pipitch_tune` | JACK tuning tool (requires JACK + FFTW3f) |
| `latency_bench` | `latency_bench` | Offline latency benchmark |

### Bundle layout

```
build/pipitch.lv2/
├── manifest.ttl
├── plugin.ttl
├── pipitch.so                  ← LV2 host loads this
├── pipitch_impl_neon.so        ← Pi 4 impl
├── pipitch_impl_armv82.so      ← Pi 5 impl
├── pipitch_ranges.conf
├── swift_f0_model.onnx
└── ModelData/
    ├── cnn_contour_model.json
    ├── cnn_note_model.json
    ├── cnn_onset_1_model.json
    ├── cnn_onset_2_model.json
    └── features_model.ort
```

### Pi 5 — manual rebuild

```bash
cd /root/pipitch_build

DEFINES="-DPIPITCH_IMPL_NAME=\"pipitch_impl_armv82\" \
  -DRTNEURAL_DEFAULT_ALIGNMENT=16 -DRTNEURAL_NAMESPACE=RTNeural \
  -DRTNEURAL_USE_EIGEN=1 -DSAVE_DOWNSAMPLED_AUDIO=0 \
  -DUSE_TEST_NOTE_FRAME_TO_TIME=0 -Dpipitch_impl_armv82_EXPORTS"
INCLUDES="-I/root/pipitch_src/LV2 -I/root/pipitch_src/NeuralNote/Lib/Model \
  -I/root/pipitch_src/NeuralNote/Lib/Utils \
  -I/root/pipitch_src/NeuralNote/ThirdParty/RTNeural \
  -I/root/pipitch_src/NeuralNote/ThirdParty/onnxruntime/include \
  -I/root/pipitch_src/NeuralNote/ThirdParty/RTNeural/RTNeural/../modules/json \
  -I/root/pipitch_src/NeuralNote/ThirdParty/RTNeural/RTNeural/.. \
  -I/root/pipitch_src/NeuralNote/ThirdParty/RTNeural/modules/Eigen"

# Rebuild impl
/usr/bin/c++ $DEFINES $INCLUDES \
  -mcpu=cortex-a76 -mtune=cortex-a76 -O3 -DNDEBUG -fPIC -O2 \
  -march=armv8.2-a+dotprod+fp16 -std=gnu++17 \
  -c /root/pipitch_src/LV2/pipitch_impl.cpp \
  -o CMakeFiles/pipitch_impl_armv82.dir/LV2/pipitch_impl.cpp.o
bash CMakeFiles/pipitch_impl_armv82.dir/link.txt

# Rebuild tune
/usr/bin/c++ -DRTNEURAL_DEFAULT_ALIGNMENT=16 -DRTNEURAL_NAMESPACE=RTNeural \
  -DRTNEURAL_USE_EIGEN=1 -DSAVE_DOWNSAMPLED_AUDIO=0 -DUSE_TEST_NOTE_FRAME_TO_TIME=0 \
  $INCLUDES \
  -mcpu=cortex-a76 -mtune=cortex-a76 -O3 -DNDEBUG -fPIE -O2 -std=gnu++17 \
  -c /root/pipitch_src/LV2/pipitch_tune.cpp \
  -o CMakeFiles/pipitch_tune.dir/LV2/pipitch_tune.cpp.o
bash CMakeFiles/pipitch_tune.dir/link.txt
```

---

## `pipitch_tune` — JACK tuning tool

A standalone JACK application that mirrors the LV2 plugin logic with detailed
console logging and a built-in synth engine for audio feedback.

```
pipitch_tune [--bundle PATH] [--config PATH]
             [--threshold 0.6] [--frame-threshold 0.5]
             [--mode poly|mono|swiftmono|swiftpoly]
             [--swift-threshold 0.5] [--provisional on|swift|none]
             [--gate 0.003] [--amp-floor 0.65]
             [--onset-blank MS] [--window MS] [--hold-cycles N]
             [--waveform sine|saw|square]
             [--attack MS] [--release MS] [--volume 0.3]
```

Run from the build directory:

```bash
cd /root/pipitch_build
./pipitch_tune --config pipitch_tune.conf --volume 1
```

---

## Deploying to Zynthian

### Paths

| Location | Path |
|----------|------|
| Pi 5 source | `/root/pipitch_src/` |
| Pi 5 build | `/root/pipitch_build/` |
| LV2 bundle (live) | `/zynthian/zynthian-plugins/lv2/pipitch.lv2/` |

### Manual deploy

```bash
cp /root/pipitch_src/LV2/plugin.ttl          /root/pipitch_build/pipitch.lv2/
cp /root/pipitch_src/LV2/pipitch_ranges.conf  /root/pipitch_build/pipitch.lv2/
cp /root/pipitch_src/LV2/pipitch_tune.conf    /root/pipitch_build/pipitch.lv2/
cp /root/pipitch_src/LV2/swift_f0_model.onnx  /root/pipitch_build/pipitch.lv2/
cp -r /root/pipitch_build/pipitch.lv2/* \
    /zynthian/zynthian-plugins/lv2/pipitch.lv2/
systemctl restart zynthian
```

---

## MIDI fan-out — multiple synth chains

**`pipitch-connect.sh`** runs at boot via `pipitch-connect.service` and:
1. Connects `PiPitch-01:midi_out → ZynMidiRouter:dev0_in` (essential)
2. Fans `ZynMidiRouter:ch0_out` out to extra synth chains listed in `FANOUT_DSTS`

Edit `FANOUT_DSTS` in the script to match your Zynthian layout.

```bash
systemctl enable  pipitch-connect.service   # persists across reboots
systemctl restart pipitch-connect.service   # re-run immediately
journalctl -u pipitch-connect.service       # view output
```

For all chains to receive MIDI simultaneously, set `ZYNTHIAN_MIDI_SINGLE_ACTIVE_CHANNEL="0"`
in `/zynthian/config/midi-profiles/default.sh`.

---

## Key constants

| Constant | Value | Meaning |
|----------|-------|---------|
| `NOTE_BASE` | 40 | Lowest MIDI note in bitmap (E2) |
| `NOTE_COUNT` | 49 | Bitmap covers MIDI 40–88 (E2–E6) |
| `PLUGIN_SR` | 22 050 Hz | Resampled rate fed to BasicPitch CNN |
| `OBP_NOTE_CAP` | 76 | OBP provisionals above E5 rejected |
| `N_CONSEC` | 4 | Consecutive agreeing OBP readings to fire |
| `ONSET_RATIO` | 3.0× | RMS fallback onset threshold |
| `ONSET_BLANK_MS` | 25 ms | Default re-trigger suppression |
| `ONSET_GATE_S` | 0.25 s | Decay-tail ghost suppression window |
| `PICK_HIGH_TIER` | 10.0 | PickDetector tier-1 ratio |
| `MPM_FFTSIZE` | 4 096 | MPM FFT size |
| `MPM_K` | 0.86 | NSDF key-maximum threshold |
| `SWIFT_POLY_KEEPALIVE` | 2 | SwiftF0 keep-alive cycles in swiftpoly |

---

## License

PiPitch LV2 plugin code is original work by Anirban Ray.
BasicPitch model and inference code from [NeuralNote](https://github.com/DamRsn/NeuralNote)
(Apache-2.0). Dependencies: [RTNeural](https://github.com/jatinchowdhury18/RTNeural) (BSD-3),
[ONNX Runtime](https://github.com/microsoft/onnxruntime) (MIT).
