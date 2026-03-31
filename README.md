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
     Two-tier: ratio >= 10 = immediate onset; 3–10 = confirmed by RMS.
   - **OBP blanking**: 5 ms freeze after PICK onset to skip pick noise.
   - **OBP** (OneBitPitchDetector): 4th-order Butterworth LP → adaptive Schmitt
     trigger → period averaging.  Requires 4 consecutive agreeing readings.
   - **HPS** (bit-parallel Harmonic Product Spectrum): cross-range OBP
     registers are shifted ×2 and ×3 to find the true fundamental.
   - **MPM** (McLeod Pitch Method, Pi 5 only): FFT autocorrelation + NSDF +
     parabolic interpolation.  Minimum 512 samples before trusting results.
   - **Confirmation buffer**: provisional held for ~10 ms before MIDI ON fires.
     If the worker corrects within that window, the correction is used instead.

   Controllable via the `provisional` parameter (on / adaptive / swift / none).

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
   | **goertzelpoly** (4) | UltraLowLatencyGoertzel | ~5 ms | Zero-latency polyphonic; NEON SIMD, Pi 5 only |

```
JACK callback (RT thread)
  ├─ PickDetector  (HPF 3 kHz → dual-EMA → transient ratio)
  ├─ OBP blanking  (5 ms freeze after PICK — skip pick noise)
  ├─ RMS onset     (fallback for hammer-ons / volume swells)
  ├─ OBP + HPS + MPM  ──────────────────────────── provisional fire
  ├─ Confirmation buffer (10 ms hold before MIDI ON)
  ├─ Ring buffer fill  (22 050 Hz resampled audio)
  ├─ Ring flush on PICK onset (zero stale audio)
  └─ Snapshot dispatch → worker thread (lockless SPSC)
       │
       ├─ [mode 0/1]  BasicPitch CNN (~95 ms)
       │     └─ buildNNBits → cancel grace → applyNotesDiff
       │
       ├─ [mode 2]    SwiftF0 (~10–20 ms)
       │     └─ resample 22050→16 kHz → infer → note lock
       │        → onset grace → ghost suppression
       │        → note-change confirmation → pitch bend snap
       │        → cancel grace → applyNotesDiff → velocity boost
       │
       ├─ [mode 3]    SwiftF0 + BasicPitch
       │     └─ SwiftF0 (~5 ms) → BasicPitch (~95 ms) → merge
       │        keep-alive bridge → active cancellation
       │        → cancel grace → applyNotesDiff
       │
       └─ [mode 4]    GoertzelPoly (audio thread, no worker)
             └─ UltraLowLatencyGoertzel: 49-bin IIR resonator bank
                onset blanking (5 ms) → multi-block eval (256 samples)
                frequency-scaled thresholds → onset ramp (200 ms)
                harmonic suppression → winner-takes-all → hold timer
```

### Provisional glitch reduction

| Technique | Description |
|-----------|------------|
| **Muted provisional** | Provisionals fire at vel 40 (~30%); boosted to 100 after SwiftF0 confirms |
| **Pitch bend snap** | ±1-3 semitone corrections use pitch bend instead of OFF+ON (no ADSR retrigger) |
| **Note lock** | Once SwiftF0 confirms, note is locked until next onset (prevents E4→silent→E4 oscillation) |
| **Confirmation buffer** | 10 ms hold before MIDI ON; worker can correct within that window |
| **OBP blanking** | 5 ms freeze after PICK skips pick noise in OBP |
| **Octave lock** | Cross-range ±12/±24 semitone suppression with onset timing gate |
| **Range priority** | In swiftMono, highest MIDI note wins across ranges |
| **Mono swap** | Immediate OFF for old notes when new note appears (no hold delay) |
| **From-silence filter** | Suppress provisionals below C3 from silence |

### 14-bit pitch bend

When `bend = on` (LV2 port 10), the PitchBendTracker provides conditional
pitch bend for natural vibrato and string bending:

| Gate | Threshold | Purpose |
|------|-----------|---------|
| Onset mask | 30 ms | No bend during attack transient |
| Stability | confidence > 0.85 for 3 frames | Only bend on stable sustained notes |
| Dead zone | ±5 cents | Keep perfectly in tune |
| Active zone | 5–100 cents | 14-bit bend (±2 semitone range) |
| Decay guard | SwiftF0 must detect same MIDI note | Snap to center when pitch drifts on decay |

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
│   ├── SwiftF0Detector.h    ← SwiftF0 ONNX wrapper (returns Hz + confidence)
│   ├── plugin.ttl / manifest.ttl
│   ├── pipitch_ranges.conf  ← Per-range config (shipped in bundle)
│   ├── pipitch_tune.conf    ← Tune tool config (includes global keys)
│   ├── pipitch_test.cpp     ← Record/test regression tool
│   ├── UltraLowLatencyGoertzel.h ← GoertzelPoly: 49-bin NEON Goertzel detector
│   ├── pipitch-connect.sh   ← JACK MIDI fan-out + synth discovery
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
| 7 | `mode` | 1 | 0–4 | Poly / Mono / SwiftMono / SwiftPoly / GoertzelPoly |
| 8 | `onset_blank_ms` | 25 | 10–100 | Re-trigger suppression (ms) |
| 9 | `provisional` | 0 | 0–3 | On / Swift / None / Adaptive |
| 10 | `bend` | 0 | 0–1 | Pitch bend Off / On |

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
| `pipitch_test` | `pipitch_test` | Record/test regression tool (requires JACK) |
| `latency_bench` | `latency_bench` | Offline latency benchmark |

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
             [--swift-threshold 0.5] [--provisional on|adaptive|swift|none|off]
             [--bend] [--octave-lock MS]
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

## MIDI routing

**`pipitch-connect.sh`** runs at boot via `pipitch-connect.service` and:
1. Connects `PiPitch-01:midi_out → ZynMidiRouter:dev0_in` (Zynthian integration)
2. Dynamically discovers all synth engine MIDI inputs
3. Connects PiPitch directly to each synth (low-latency bypass)

```bash
systemctl enable  pipitch-connect.service   # persists across reboots
systemctl restart pipitch-connect.service   # re-run immediately
journalctl -u pipitch-connect.service       # view output
```

For all chains to receive MIDI simultaneously, set `ZYNTHIAN_MIDI_SINGLE_ACTIVE_CHANNEL="0"`
in `/zynthian/config/midi-profiles/default.sh`.

---

## Configuration keys

### Global (pipitch_tune.conf / CLI)

| Key | CLI | Default | Description |
|-----|-----|---------|-------------|
| `gate_floor` | `--gate` | 0.003 | Noise gate floor |
| `amp_floor` | `--amp-floor` | 0.65 | BasicPitch amplitude floor |
| `threshold` | `--threshold` | 0.6 | Onset sensitivity |
| `frame_threshold` | `--frame-threshold` | 0.5 | Per-frame CNN confidence |
| `mode` | `--mode` | mono | poly / mono / swiftmono / swiftpoly / goertzelpoly |
| `provisional` | `--provisional` | on | on / adaptive / swift / none / off |
| `onset_blank_ms` | `--onset-blank` | 25 | Re-trigger suppression (ms) |
| `swift_threshold` | `--swift-threshold` | 0.5 | SwiftF0 confidence threshold |
| `octave_lock_ms` | `--octave-lock` | 250 | Octave jump suppression window (ms) |
| `bend` | `--bend` | off | Enable 14-bit pitch bend |

### Per-range (pipitch_ranges.conf)

| Key | Default | Description |
|-----|---------|-------------|
| `name` | — | Range label |
| `midi_low` | — | Lowest MIDI note (inclusive) |
| `midi_high` | — | Highest MIDI note (inclusive) |
| `window` | 150 | CNN capture window (ms) |
| `min_note_length` | 6 | Minimum CNN frames |
| `hold_cycles` | 2 | Inference cycles before note-OFF |
| `swift_hold_cycles` | 2 | Hold cycles for SwiftF0 |

---

## Key constants

| Constant | Value | Meaning |
|----------|-------|---------|
| `NOTE_BASE` | 40 | Lowest MIDI note in bitmap (E2) |
| `NOTE_COUNT` | 49 | Bitmap covers MIDI 40–88 (E2–E6) |
| `PLUGIN_SR` | 22 050 Hz | Resampled rate fed to BasicPitch CNN |
| `OBP_NOTE_CAP` | 76 | OBP provisionals above E5 rejected |
| `N_CONSEC` | 4 | Consecutive agreeing OBP readings to fire |
| `MPM_FFTSIZE` | 4 096 | MPM FFT size |
| `MPM_K` | 0.86 | NSDF key-maximum threshold |
| `SWIFT_POLY_KEEPALIVE` | 2 | SwiftF0 keep-alive cycles in swiftpoly |

---

## GoertzelPoly mode (mode 4)

Zero-latency polyphonic pitch detection using a 49-bin Goertzel IIR resonator
bank running entirely in the audio thread (no worker needed).  Pi 5 only
(requires AArch64 NEON SIMD).

### Signal processing pipeline

1. **Onset blanking** (5 ms): After a pick onset, note detection freezes while
   the IIR filters continue running.  Prevents broadband transient from
   triggering all 49 bins.

2. **Multi-block evaluation** (256 samples / ~5.3 ms): IIR processes
   sample-by-sample, but magnitudes and note decisions are computed every 256
   samples — enough for the Goertzel to resolve guitar fundamentals.

3. **Frequency-scaled thresholds**: Low-frequency bins (E2 = 82 Hz) require
   16× higher magnitude than mid-range bins (E4 = 330 Hz).  Quadratic scaling:
   `threshold = base × (330/freq)²`.  Eliminates low-frequency ringing from
   mid-range attacks.

4. **Onset-aware dynamic threshold** (200 ms ramp): ON threshold elevated up to
   20× immediately after onset, ramping linearly back to normal.  Rejects
   residual broadband energy while allowing the true fundamental to grow.

5. **Harmonic suppression**: Checks against raw (pre-suppression) magnitudes;
   suppresses octave, fifth, third, fourth, and higher harmonics when a lower
   potential fundamental is present.  Minimum magnitude floor of 0.1 to avoid
   noise-floor artifacts.

6. **Winner-takes-all**: Within each 12-note octave window, only the strongest
   bin survives.

7. **Hold timer** (6 eval cycles / ~30 ms): Once a note turns ON, it stays ON
   for at least 30 ms even if magnitude briefly dips.  Prevents ON/OFF
   flickering from magnitude oscillation near threshold.

8. **Octave-lock**: At the consumer level, new Goertzel note-ONs are suppressed
   if ±12/±24 semitones from an already-active note.

### Limitations

- Provisionals (OBP+MPM) are disabled in GoertzelPoly mode
- The ring buffer / CNN / SwiftF0 pipeline is bypassed entirely
- Best suited for monophonic playing; chord detection still noisy

---

## `pipitch_test` — record & regression test tool

A standalone tool for recording guitar audio and testing PiPitch detection
accuracy against labeled note sequences.

### Record mode

Capture live audio from JACK and save to WAV on Ctrl+C:

```bash
pipitch_test record -o guitar_e4.wav [--port system:capture_1]
```

### Test mode

Feed a WAV file through the full PiPitch pipeline and compare detected notes
against a label file:

```bash
pipitch_test test -i guitar_e4.wav -l labels.txt \
    [--mode goertzelpoly] [--config pipitch_tune.conf]
```

### Label file format

**Mono** — one note per line (played in sequence):

```
mono
E4
D4
F#4
```

**Chord** — one chord per line (played in sequence):

```
chord
Em
G
Am
```

Supported chord qualities: major, minor (`m`), 7th (`7`), minor 7th (`m7`),
major 7th (`maj7`), diminished (`dim`), augmented (`aug`), sus2, sus4,
power (`5`).

### Scoring

The tool segments detected MIDI events by silence gaps and matches segments to
labels in order.  Reports hits, partial hits, misses, wrong notes, and
extra/missing segments.

---

## License

PiPitch LV2 plugin code is original work by Anirban Ray.
BasicPitch model and inference code from [NeuralNote](https://github.com/DamRsn/NeuralNote)
(Apache-2.0). Dependencies: [RTNeural](https://github.com/jatinchowdhury18/RTNeural) (BSD-3),
[ONNX Runtime](https://github.com/microsoft/onnxruntime) (MIT).
