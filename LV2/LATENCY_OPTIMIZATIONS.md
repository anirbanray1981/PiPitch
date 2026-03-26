# NeuralNote Guitar2MIDI — Latency Optimization Notes

## Current Latency Model

```
Pick note
   │
   ▼
[JACK buffer]           ~3 ms       (jack_get_buffer_size / sample_rate)
   │
   ▼
[Ring buffer fills]     window ms   audio must accumulate before inference can run
   │
   ▼
[Dispatch wait]         window/2 ms freshSamples threshold — half-window wait before
   │                                queuing a snapshot to the worker
   ▼
[CNN inference]         ~50–100 ms  BasicPitch transcribeToMIDI() on Pi5
   │                                (faster with shorter windows)
   ▼
MIDI note-ON
```

For a 150 ms window on Pi5:
`~3 ms + ~75 ms dispatch wait + ~95 ms inference ≈ 173 ms total`

---

## Optimizations

### 1. Lower the dispatch threshold
**Effort:** trivial — one-line change
**Gain:** ~30 ms
**Files:** `neuralnote_impl.cpp`, `neuralnote_tune.cpp`

Currently a new snapshot is only sent to the worker once `freshSamples >= ringSize / 2`,
introducing a built-in half-window wait. Lowering this dispatches more often on slightly
staler audio.

```cpp
// Current (both files, in the snapshot-dispatch block):
if (r.ringFilled >= r.ringSize && r.freshSamples >= r.ringSize / 2)

// Change to a fixed minimum — e.g. 25 ms of resampled audio:
static constexpr int MIN_FRESH_NEON = static_cast<int>(AUDIO_SAMPLE_RATE * 0.025f); // neuralnote_impl
static constexpr int MIN_FRESH_TUNE = static_cast<int>(PLUGIN_SR          * 0.025f); // neuralnote_tune

if (r.ringFilled >= r.ringSize && r.freshSamples >= MIN_FRESH_*)
```

**Trade-off:** inference runs 4–6× per window instead of 2×; CPU use roughly doubles.
On Pi5 this is still well within budget.

---

### 2. Tune window sizes down per range
**Effort:** none — already implemented via per-range config
**Gain:** variable (10–80 ms depending on range)
**Files:** `neuralnote_tune.conf` / `neuralnote_ranges.conf`

Shorter windows = less fill time and faster inference. The practical floor is ~35 ms
(3 CQT frames) but accuracy degrades below ~60 ms until the circular state warms up.
High strings (C5+) can use 60 ms; low strings (E2–B2) benefit from 120–150 ms.

See `neuralnote_tune.conf.sample` for a starting point.

---

### 3. Per-range parallel worker threads
**Effort:** moderate — refactor worker thread per RangeState
**Gain:** scales with number of ranges and CPU cores (Pi5 has 4 cores)
**Files:** `neuralnote_impl.cpp`, `neuralnote_tune.cpp`

Currently all ranges run sequentially in one worker thread. Total inference time =
sum of all range inference times. With one thread per range, it becomes max of all
range inference times — a 4-range config could run 4× faster.

**Implementation sketch:**
- Move `workerThread`, `workerMutex`, `workerCv`, `workerQuit` into `PerRangeState` /
  `RangeState`.
- Each range's thread blocks on its own cv, processes only its own snapshot, writes
  to the shared `pendingMidi` / `pendingNotes` (still needs a mutex on that).
- `run()` / `jackProcess()` notifies each range's cv independently.

**Trade-off:** more threads, but Pi5 has spare cores and inference is compute-bound.

---

### 4. Onset-triggered early dispatch
**Effort:** significant — new onset detector component
**Gain:** ~50–75 ms (eliminates most of the dispatch wait for note attacks)
**Files:** new `OnsetDetector.h`, `neuralnote_impl.cpp`, `neuralnote_tune.cpp`

A simple energy / spectral-flux onset detector can fire within 5 ms of a pick attack.
When an onset is detected, immediately dispatch the current ring to the worker without
waiting for `freshSamples` to reach the threshold.

```
pick → onset detected (5 ms) → immediate dispatch → inference (~95 ms) → MIDI on
       vs. normal path:         wait ~75 ms        → inference (~95 ms) → MIDI on
```

**Implementation sketch:**
```cpp
// In run() / jackProcess() after pushing to ring:
if (detectOnset(self->audioIn, nSamples, gateFloor)) {
    // Force dispatch immediately for all ranges whose ring is full
    // (ignore freshSamples threshold)
    forceDispatch(self);
}
```

Simple onset detector (no external deps):
- Compute block RMS; compare to a smoothed background level.
- If `rms > background * onset_ratio` (e.g. 3.0×), trigger onset.
- Blank for ~50 ms after trigger to avoid re-triggering on the same note.

**Trade-off:** can false-trigger on loud sustained notes or percussive noise.
Tune `onset_ratio` and blank time carefully.

---

### 5. Two-phase pitch detection (YIN + CNN)
**Effort:** complex — new pitch detector, note reconciliation logic
**Gain:** ~100 ms perceived latency reduction for note attacks
**Files:** new `YinPitchDetector.h`, significant changes to impl/tune

Fire a provisional note-ON immediately using the YIN autocorrelation algorithm
(runs in ~1 ms on a short audio frame), then confirm / correct with BasicPitch.

```
pick → YIN (~1 ms)          → provisional note-ON sent immediately
     → BasicPitch (~95 ms)  → if pitch differs: send note-OFF + corrected note-ON
                             → if pitch matches: do nothing (note already on)
```

This is how commercial guitar-to-MIDI systems (Roland GR series) achieve low
perceived latency while maintaining pitch accuracy.

**Implementation sketch:**
1. After each JACK block, run YIN on the last ~50 ms of audio.
2. If a clear pitch is detected and no note is currently active in that range:
   - Send provisional note-ON for the detected MIDI note.
   - Flag that range as "awaiting CNN confirmation".
3. When BasicPitch inference completes for that range:
   - If the detected pitch matches: do nothing.
   - If it differs: send note-OFF for provisional pitch, note-ON for CNN pitch.
   - If no note found: send note-OFF for provisional pitch.

**Trade-off:** occasional brief wrong-pitch notes during the CNN confirmation window
(~95 ms). Acceptable for monophonic lines; less ideal for fast chords.

YIN reference implementation: de Cheveigné & Kawahara (2002).
A minimal C++ implementation is ~100 lines.

---

## Recommended Implementation Order

1. **Lower dispatch threshold** — trivial, immediate ~30 ms gain, do this first.
2. **Tune window sizes** — already done; keep refining with `neuralnote_tune`.
3. **Parallel per-range workers** — moderate effort, scales well on Pi5's 4 cores.
4. **Onset-triggered dispatch** — significant gain for attack latency, moderate risk.
5. **Two-phase YIN + CNN** — highest perceived gain, most complex, do last.

Combined effect of all five: estimated total latency **< 50 ms** for note attacks
on Pi5, with ~95 ms steady-state for held notes.
