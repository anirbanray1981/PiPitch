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
     registers are shifted x2 and x3 to find the true fundamental.
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
   | **swiftmono** (2) | SwiftF0 ONNX | ~10–20 ms | Monophonic; 389 KB model, 16 kHz input; supports pitch bend |
   | **swiftpoly** (3) | SwiftF0 + BasicPitch | ~100 ms | SwiftF0 for fast note-ON, BasicPitch for sustain/OFF |
   | **goertzelmono** (4) | UltraLowLatencyGoertzel | ~5 ms | Zero-latency monophonic; NEON SIMD, Pi 5 only (default) |
   | **goertzelpoly** (5) | Goertzel + BasicPitch | ~5 ms + ~95 ms | Goertzel scout + CNN judge; polyphonic chord detection |

```
JACK callback (RT thread)
  |-- PickDetector  (HPF 3 kHz -> dual-EMA -> transient ratio)
  |-- OBP blanking  (5 ms freeze after PICK -- skip pick noise)
  |-- RMS onset     (fallback for hammer-ons / volume swells)
  |-- OBP + HPS + MPM  ----------------------- provisional fire
  |-- Confirmation buffer (10 ms hold before MIDI ON)
  |-- Ring buffer fill  (22 050 Hz resampled audio)
  |-- Ring flush on PICK onset (zero stale audio)
  +-- Snapshot dispatch -> worker thread (lockless SPSC)
       |
       |-- [mode 0/1]  BasicPitch CNN (~95 ms)
       |     +-- buildNNBits -> cancel grace -> applyNotesDiff
       |
       |-- [mode 2]    SwiftF0 (~10-20 ms)
       |     +-- resample 22050->16 kHz -> infer -> note lock
       |        -> onset grace -> ghost suppression
       |        -> note-change confirmation -> pitch bend snap
       |        -> cancel grace -> applyNotesDiff -> velocity boost
       |
       |-- [mode 3]    SwiftF0 + BasicPitch
       |     +-- SwiftF0 (~5 ms) -> BasicPitch (~95 ms) -> merge
       |        keep-alive bridge -> active cancellation
       |        -> cancel grace -> applyNotesDiff
       |
       |-- [mode 4]    GoertzelMono (audio thread, no worker)
       |     +-- UltraLowLatencyGoertzel: 49-bin IIR resonator bank
       |        onset blanking (5 ms) -> multi-block eval (192 samples)
       |        frequency-scaled thresholds -> onset ramp (50 ms)
       |        harmonic suppression -> winner-takes-all (incumbent 3x)
       |        -> onset-gated note-ON -> pitch snap (+/-2 semitone bend)
       |
       +-- [mode 5]    GoertzelPoly (audio thread + worker)
             +-- Audio thread: Goertzel scout (muted vel 40, up to max_poly)
                 no octave-lock, no onset quench, 50 ms strum window
             +-- Worker: BasicPitch CNN judge (~95 ms)
                 confirm (boost vel) / add (CNN-only notes) / veto (harmonics)
                 hold-before-veto: holdCycles consecutive CNN misses required
```

### Provisional glitch reduction

| Technique | Description |
|-----------|------------|
| **Muted provisional** | Provisionals fire at vel 40 (~30%); boosted to 100 after SwiftF0 confirms |
| **Pitch bend snap** | +/-1-3 semitone corrections use pitch bend instead of OFF+ON (no ADSR retrigger) |
| **Note lock** | Once SwiftF0 confirms, note is locked until next onset (prevents E4->silent->E4 oscillation) |
| **Confirmation buffer** | 10 ms hold before MIDI ON; worker can correct within that window |
| **OBP blanking** | 5 ms freeze after PICK skips pick noise in OBP |
| **Octave lock** | Cross-range +/-12/+/-24 semitone suppression with onset timing gate |
| **Range priority** | In swiftMono, highest MIDI note wins across ranges |
| **Mono swap** | Immediate OFF for old notes when new note appears (no hold delay) |
| **From-silence filter** | Suppress provisionals below C3 from silence |

### 14-bit pitch bend (swiftmono)

When `bend = on` (LV2 port 10), the PitchBendTracker provides continuous
14-bit MIDI pitch bend for natural vibrato and string bending.  Available in
**swiftmono mode only** — SwiftF0's per-frame Hz output provides the
sub-semitone resolution needed for accurate bend tracking.

| Gate | Threshold | Purpose |
|------|-----------|---------|
| Onset mask | 30 ms | No bend during attack transient |
| Stability | confidence > 0.85 for 3 frames | Only bend on stable sustained notes |
| Dead zone | +/-5 cents | Keep perfectly in tune |
| Active zone | 5–100 cents | 14-bit bend (+/-2 semitone range) |
| Decay guard | SwiftF0 must detect same MIDI note | Snap to center when pitch drifts on decay |

### GoertzelPoly — "Fast Scout & Wise Judge" (mode 5)

GoertzelPoly combines Goertzel's zero-latency onset detection with BasicPitch
CNN's polyphonic accuracy for chord detection.

**Audio thread — Goertzel scout:**
- On each onset, opens a 50 ms window accepting up to `max_poly` notes
  (default 3).  Subsequent onsets within the window extend it without
  resetting the note count, allowing chord strums to accumulate.
- Note-ONs fire at muted velocity (40) so harmonic ghosts are barely audible.
- No octave-lock — all candidates pass through for CNN to judge.
- No onset quench — in poly mode, new string hits must not cancel
  previously-detected chord tones.
- Ring buffer is fed simultaneously for CNN analysis.

**Worker thread — BasicPitch CNN judge:**
- Runs CNN inference (~95 ms) on ring-buffer snapshots.
- **Confirm**: notes seen by both CNN and Goertzel get velocity boost to 100.
- **Add**: notes CNN detects that Goertzel missed are sent at full velocity.
- **Veto**: notes Goertzel has but CNN doesn't see (harmonic ghosts) are
  removed — but only after both Goertzel's IIR AND CNN drop the note for
  `holdCycles` consecutive inference cycles.  While Goertzel still detects
  the note, it stays alive regardless of CNN.
- Each range only processes notes within its `[midiLow, midiHigh]` boundaries.

**Recommended thresholds for chord detection:**
`amp_floor = 0.3`, `frame_threshold = 0.4`.  The defaults are tuned for this.
Higher values cause CNN to miss quieter chord tones.

### Latency benchmarks (Pi 5, synthetic guitar, 48 kHz)

Measured using `pipitch_test` with synthetic guitar WAV files (harmonics +
pick transient + exponential decay).  Latency = time from note onset to first
MIDI ON event.  "+P" = provisional detection on, "-P" = provisional off.

| Note | poly +P | poly -P | mono +P | mono -P | swift +P | swift -P | GoertzelMono | GoertzelPoly |
|------|---------|---------|---------|---------|----------|----------|-------------|-------------|
| E2 (82 Hz) | 144 ms | 147 ms | 147 ms | 148 ms | 139 ms | 136 ms | **59 ms** | 147 ms |
| E3 (165 Hz) | -- | 97 ms | -- | 95 ms | **76 ms** | 125 ms | **43 ms** | **43 ms** |
| E4 (330 Hz) | -- | 172 ms | -- | 173 ms | **83 ms** | 136 ms | **35 ms** | **35 ms** |
| E5 (659 Hz) | -- | 185 ms | -- | 181 ms | **68 ms** | 116 ms | **15 ms** | **15 ms** |
| E6 (1319 Hz) | 128 ms | 131 ms | 132 ms | 133 ms | 125 ms | 128 ms | **15 ms** | **15 ms** |

**Key observations:**

- **GoertzelMono is fastest** across all notes (15--59 ms).  Runs entirely in
  the audio thread with no worker latency.
- **GoertzelPoly matches GoertzelMono** for E3--E6 (Goertzel scout fires in
  audio thread).  E2 is slower (147 ms) due to the 14-cycle confidence buffer
  for very low notes.
- **SwiftF0 with provisional** is the fastest worker-based mode (68--139 ms).
  Provisional fires before SwiftF0 inference completes.
- **Poly/Mono provisional** can fail on synthetic waveforms ("--") where
  OBP+HPS+MPM consensus isn't reached.  CNN-only latency is 95--185 ms.
- **Higher notes = lower latency** in all modes: fewer IIR cycles to resolve
  the fundamental, fewer OBP readings for consensus.

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
  GoertzelMono and GoertzelPoly also run Goertzel IIR processing here.
- **Worker thread** (one, shared across all ranges): inference, `applyNotesDiff`, MIDI output.
  GoertzelMono bypasses the worker entirely; GoertzelPoly uses it for CNN confirmation.
- All comms are lockless: `SnapshotChannel` (SPSC atomic + semaphore), `MidiOutQueue` (SPSC ring).

---

## Repository structure

```
PiPitch/
+-- NeuralNote/              <-- git submodule (BasicPitch model code + dependencies)
|   +-- Lib/Model/           <-- BasicPitch CNN inference
|   +-- ThirdParty/RTNeural/ <-- Neural network runtime
|   +-- ThirdParty/onnxruntime/
+-- LV2/                     <-- PiPitch plugin code
|   +-- pipitch_impl.cpp     <-- LV2 plugin (RT callback + ImplWorkerHooks)
|   +-- pipitch_tune.cpp     <-- JACK tuning tool (TuneWorkerHooks + synth)
|   +-- pipitch.cpp           <-- LV2 wrapper (CPU dispatch via dlopen)
|   +-- PiPitchShared.h      <-- Shared: constants, pipeline, runWorkerCommon<Hooks>
|   +-- SwiftF0Detector.h    <-- SwiftF0 ONNX wrapper (returns Hz + confidence)
|   +-- UltraLowLatencyGoertzel.h <-- Goertzel: 49-bin NEON IIR resonator bank
|   +-- plugin.ttl / manifest.ttl
|   +-- pipitch_ranges.conf  <-- Per-range config (shipped in bundle)
|   +-- pipitch_tune.conf    <-- Tune tool config (includes global keys)
|   +-- pipitch_test.cpp     <-- Record/test regression tool
|   +-- pipitch-connect.sh   <-- JACK MIDI fan-out + synth discovery
|   +-- pipitch-connect.service
+-- CMakeLists.txt           <-- LV2 build (references NeuralNote/ submodule)
+-- README.md
```

---

## LV2 ports

| Index | Symbol | Default | Range | Description |
|-------|--------|---------|-------|-------------|
| 0 | `audio_in` | -- | -- | Mono audio in |
| 1 | `midi_out` | -- | -- | MIDI output (atom sequence) |
| 2 | `audio_out` | -- | -- | Audio through |
| 3 | `threshold` | 0.6 | 0.1–1.0 | Onset sensitivity |
| 4 | `gate_floor` | 0.003 | 0.0–0.1 | Noise gate floor |
| 5 | `amp_floor` | 0.3 | 0.0–1.0 | BasicPitch amplitude floor |
| 6 | `frame_threshold` | 0.4 | 0.05–0.95 | Per-frame CNN confidence |
| 7 | `mode` | 4 | 0–5 | Poly / Mono / SwiftMono / SwiftPoly / GoertzelMono / GoertzelPoly |
| 8 | `onset_blank_ms` | 25 | 10–100 | Re-trigger suppression (ms) |
| 9 | `provisional` | 0 | 0–3 | On / Swift / None / Adaptive |
| 10 | `bend` | 0 | 0–1 | Pitch bend Off / On (swiftmono only) |
| 11 | `max_poly` | 3 | 1–6 | Max simultaneous notes in GoertzelPoly |

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
| `PiPitchImpl_NEON` | `pipitch_impl_neon.so` | Pi 4 (ARMv8-A, NEON) -- no MPM |
| `PiPitchImpl_ARMv82` | `pipitch_impl_armv82.so` | Pi 5 (ARMv8.2-A, dotprod+fp16) -- MPM enabled |
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
             [--threshold 0.6] [--frame-threshold 0.4]
             [--mode poly|mono|swiftmono|swiftpoly|goertzelmono|goertzelpoly]
             [--swift-threshold 0.5] [--provisional on|adaptive|swift|none|off]
             [--bend] [--octave-lock MS] [--max-poly N]
             [--gate 0.003] [--amp-floor 0.3]
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
1. Connects `PiPitch-01:midi_out -> ZynMidiRouter:dev0_in` (Zynthian integration)
2. Dynamically discovers all synth engine MIDI inputs
3. Connects PiPitch directly to each synth (low-latency bypass)

```bash
systemctl enable  pipitch-connect.service   # persists across reboots
systemctl restart pipitch-connect.service   # re-run immediately
journalctl -u pipitch-connect.service       # view output
```

For all chains to receive MIDI simultaneously, set `ZYNTHIAN_MIDI_SINGLE_ACTIVE_CHANNEL="0"`
in `/zynthian/config/midi-profiles/default.sh`.

**Note:** GoertzelMono and GoertzelPoly modes output on MIDI channel 2.
Configure your synth chain to listen on channel 2 accordingly.

---

## Configuration keys

### Global (pipitch_tune.conf / CLI)

| Key | CLI | Default | Description |
|-----|-----|---------|-------------|
| `gate_floor` | `--gate` | 0.003 | Noise gate floor |
| `amp_floor` | `--amp-floor` | 0.3 | BasicPitch amplitude floor |
| `threshold` | `--threshold` | 0.6 | Onset sensitivity |
| `frame_threshold` | `--frame-threshold` | 0.4 | Per-frame CNN confidence |
| `mode` | `--mode` | goertzelmono | poly / mono / swiftmono / swiftpoly / goertzelmono / goertzelpoly |
| `provisional` | `--provisional` | on | on / adaptive / swift / none / off |
| `onset_blank_ms` | `--onset-blank` | 25 | Re-trigger suppression (ms) |
| `swift_threshold` | `--swift-threshold` | 0.5 | SwiftF0 confidence threshold |
| `octave_lock_ms` | `--octave-lock` | 250 | Octave jump suppression window (ms) |
| `bend` | `--bend` | off | Enable 14-bit pitch bend (swiftmono only) |
| `max_poly` | `--max-poly` | 3 | Max simultaneous Goertzel notes per onset (goertzelpoly) |

### Per-range (pipitch_ranges.conf)

| Key | Default | Description |
|-----|---------|-------------|
| `name` | -- | Range label |
| `midi_low` | -- | Lowest MIDI note (inclusive) |
| `midi_high` | -- | Highest MIDI note (inclusive) |
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

## GoertzelMono mode (mode 4) — default

Zero-latency monophonic pitch detection using a 49-bin Goertzel IIR resonator
bank running entirely in the audio thread (no worker needed).  Pi 5 only
(requires AArch64 NEON SIMD).

### Signal processing pipeline

1. **Onset blanking** (5 ms): After a pick onset, note detection freezes while
   the IIR filters continue running.  Prevents broadband transient from
   triggering all 49 bins.

2. **Multi-block evaluation** (192 samples / ~4 ms): IIR processes
   sample-by-sample, but magnitudes and note decisions are computed every 192
   samples -- enough for the Goertzel to resolve guitar fundamentals.

3. **Frequency-scaled thresholds**: Low-frequency bins (E2 = 82 Hz) require
   ~4x higher magnitude than mid-range bins (E4 = 330 Hz).  Linear scaling:
   `threshold = base x (330/freq)`.

4. **Onset-aware dynamic threshold** (50 ms ramp): ON threshold elevated up to
   8x immediately after onset, ramping linearly back to normal.  Rejects
   residual broadband energy while allowing the true fundamental to grow.

5. **Harmonic suppression**: Checks against raw (pre-suppression) magnitudes;
   suppresses octave (+/-1 semitone tolerance), fifth, third, fourth, and higher
   harmonics when a lower potential fundamental is present.  Minimum magnitude
   floor of 0.1 to avoid noise-floor artifacts.

6. **Winner-takes-all with incumbent advantage**: Within each 12-note octave
   window, only the strongest bin survives.  An already-ON note needs the
   competitor to be 3x stronger to be dethroned -- prevents decay-phase
   toggling between adjacent semitones.

7. **Hold timer** (8 eval cycles / ~32 ms): Once a note turns ON, it stays ON
   for at least 32 ms even if magnitude briefly dips.  Prevents ON/OFF
   flickering from magnitude oscillation near threshold.

8. **Onset-gated note-ON**: New note-ONs are only allowed within a 250 ms
   window after a pick/RMS onset.  The first note to fire closes the window
   (one note per onset).  This single rule replaces all harmonic guard logic --
   no onset means no new notes, so decay-phase ghosts, sub-harmonics, and
   adjacent-semitone leakage are all suppressed.

9. **Onset quench**: On each onset, IIR state (s1/s2) of all active notes is
   halved and their activeCount/holdRemain reset.  This clears stale energy
   so the new note can build confidence without competing against residual
   old-note energy.  Disabled in GoertzelPoly mode (chord tones must accumulate).

10. **Tiered confidence**: Low notes need more eval cycles before firing to
    avoid spectral leakage false triggers.  E2: 14 cycles (~56 ms),
    F2--A#2: 10 (~40 ms), B2--E4: 6 (~24 ms), F4+: 3 (~12 ms).

11. **Pitch snap**: When Goertzel detects a note change within +/-2 semitones of
    the currently-sounding note, a MIDI pitch bend is sent instead of OFF+ON.
    This avoids synth ADSR re-trigger glitches.  The bend is centered on the
    next pick onset or when the note turns off.

12. **Octave-lock**: New Goertzel note-ONs are suppressed if +/-12/+/-24
    semitones from an already-active note.

13. **Minimum velocity** (25): Goertzel detections below velocity 25 are
    rejected as noise-floor artifacts.

### MIDI channel

GoertzelMono and GoertzelPoly output on MIDI channel 2 (0-indexed: 1) to
avoid double-triggering when routed through both ZynMidiRouter and direct
synth connections.

### Limitations

- Provisionals (OBP+MPM) are disabled in GoertzelMono mode
- The ring buffer / CNN / SwiftF0 pipeline is bypassed entirely
- Best suited for monophonic playing; use GoertzelPoly for chords

---

## GoertzelPoly mode (mode 5)

Hybrid polyphonic detection: Goertzel provides fast scout note-ONs, BasicPitch
CNN confirms, adds missed chord tones, or vetoes harmonic ghosts.

### Architecture: "Fast Scout & Wise Judge"

**Audio thread -- Goertzel scout (< 5 ms):**
- On onset, opens a 50 ms window accepting up to `max_poly` notes (default 3)
- Subsequent onsets within the window extend it (strum accumulation)
- Note-ONs fire at muted velocity (40) -- ghosts are barely audible
- No octave-lock (all candidates pass through for CNN)
- No onset quench (chord tones must accumulate, not cancel each other)
- Ring buffer fed simultaneously for CNN analysis

**Worker thread -- BasicPitch CNN judge (~95 ms):**
- **Confirm**: CNN + Goertzel agree -> velocity boost to 100
- **Add**: CNN detects notes Goertzel missed -> note-ON at full velocity
- **Veto**: Goertzel has notes CNN doesn't see (harmonic ghosts) -> note-OFF,
  but only after both Goertzel IIR AND CNN drop the note for `holdCycles`
  consecutive cycles.  While Goertzel still detects the note, it stays alive.
- Each range only processes notes within its `[midiLow, midiHigh]` boundaries

### Tuning

Recommended thresholds: `amp_floor = 0.3`, `frame_threshold = 0.4` (the
defaults).  Higher values cause CNN to miss quieter chord tones.  The sweep
results for a Bm chord (B3, D4, F#4):

| amp_floor | frame_thr | Chord hit | Wrong notes |
|-----------|-----------|-----------|-------------|
| 0.2 | 0.4 | 100% | 2 (octave harmonics) |
| 0.3 | 0.4 | 100% | 2 |
| 0.4+ | any | 0% (partial) | 0 |
| 0.65 | any | 0% (partial) | 0 |

---

## `pipitch_test` -- record & regression test tool

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
    [--mode goertzelpoly] [--max-poly 3] [--config pipitch_tune.conf]
```

### Label file format

**Mono** -- one note per line (played in sequence):

```
mono
E4
D4
F#4
```

**Chord** -- one chord per line (played in sequence):

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
