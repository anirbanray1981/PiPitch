# NeuralNote Guitar2MIDI — Latency Optimization Notes

## Current Architecture (two-phase)

```
Pick attack
   │
   ├─ Onset detector fires (1 JACK buffer ≈ 3–6 ms)
   │      │
   │      ├─ OBP + HPS: vote in 4×16 = 64 native-SR samples (~1.5 ms)
   │      └─ MPM fill: waits for enough samples → result only when fill sufficient
   │                   (may suppress provisional until MPM buffer fills)
   │
   ├─ [Phase 1] Provisional note-ON  ← OBP + HPS + MPM agree
   │      fired in same JACK callback as OBP vote; MIDI to LV2 atom / synth immediately
   │
   └─ [Phase 2] CNN confirmation / correction / cancel  (~95 ms on Pi5)
          snapshot dispatched on onset (MIN_FRESH_FLOOR = 25 ms threshold)
          → worker runs BasicPitch → midiOut queue → drain next run()
```

### Provisional latency budget (LV2 plugin)
| Stage | Time |
|---|---|
| JACK buffer (onset detection granularity) | 3–6 ms |
| OBP consensus (4 × 16 samples) | ~1.5 ms |
| MPM fill wait | **dominant — see item 1** |
| LV2 atom → downstream synth | 0 ms (same JACK cycle if graph order correct) |
| **Total (best case)** | **~5–10 ms** |

### CNN confirmation latency budget
| Stage | Time |
|---|---|
| Onset → snapshot dispatch | ~25 ms (MIN_FRESH_FLOOR) |
| BasicPitch inference on Pi5 | ~95 ms |
| midiOut queue → next run() drain | 1 JACK buffer |
| **Total** | **~120–125 ms** |

---

## Known LV2 vs neuralnote_tune Latency Gap

The following differences explain why the LV2 plugin feels later than neuralnote_tune:

| Source | Tune | LV2 plugin |
|---|---|---|
| Provisional audio start | Synth voice starts at sample 0 of **same** callback | MIDI atom → downstream synth in same cycle (if graph order correct) |
| CNN note drain | `processSynth` at **end** of callback | `midiOut` drain at **start** of **next** callback |
| JACK buffer size | User-controlled (often 64–128) | Zynthian default (often 256) |

---

## Pending Optimizations

### 1. ~~MPM fallback — fire on OBP+HPS if MPM not ready~~ ❌ DO NOT IMPLEMENT

**Tried and reverted** — caused ghost notes in practice.

OBP+HPS without MPM confirmation fires on harmonics and string resonance that MPM
correctly rejects. The CNN blacklist and correction mechanism is not fast enough to
suppress these before they become audible. MPM "not ready" suppression must be kept.

---

### 2. Double-drain `midiOut` in `run()` — reduce CNN note latency by one JACK buffer
**Effort:** trivial — duplicate the drain loop at the end of `run()`
**Gain:** up to one JACK buffer (~3–6 ms) for CNN-confirmed notes
**Files:** `neuralnote_impl.cpp`

Currently CNN-confirmed notes pushed by the worker during a callback are not visible
until the drain at the top of the **next** callback. Adding a second drain at the end
of `run()` (after audio processing) catches any events pushed during this cycle.

```cpp
// At the very end of run(), before lv2_atom_forge_pop():
for (auto& rp : self->ranges) {
    PendingNote pn;
    while (rp->midiOut.pop(pn))
        writeMidi(&self->forge, 0, self->uris.midi_MidiEvent,
                  pn.noteOn ? uint8_t(0x90) : uint8_t(0x80),
                  static_cast<uint8_t>(pn.pitch),
                  pn.noteOn ? static_cast<uint8_t>(pn.velocity) : uint8_t(0));
}
```

**Trade-off:** none — purely additive. Some notes may be drained twice (once at start,
once at end), but the SPSC queue guarantees each event is popped exactly once.

---

### 3. JACK graph processing order in Zynthian
**Effort:** zero code — Zynthian configuration only
**Gain:** up to one full JACK buffer if currently mis-ordered

If Zynthian schedules the synthesizer **before** NeuralNote in its processing graph,
MIDI events written to the LV2 atom output this cycle won't reach the synth until the
next cycle — adding one full JACK buffer of latency.

**Fix:** verify in Zynthian's engine/JACK connection view that NeuralNote feeds into the
synth, not the other way around. In JACK terms: NeuralNote's output port should connect
to the synth's input port, and NeuralNote should appear earlier in `jack_lsp -c` output.

---

### 4. JACK buffer size
**Effort:** zero code — system configuration
**Gain:** 3–8 ms per halving of buffer size

Zynthian defaults to 256 samples at 44100 Hz ≈ 5.8 ms per callback. neuralnote_tune
is often tested at 128 or 64 samples. Each extra sample delays onset detection
granularity by the same amount. Reduce in Zynthian → System → Audio settings.

**Trade-off:** smaller buffers increase CPU overhead and risk of JACK xruns. 128 is
usually stable on Pi5; 64 may require real-time kernel tuning.

---

### 5. MIDI frame timestamp accuracy
**Effort:** small — compute onset sample offset in run()
**Gain:** up to half a JACK buffer of perceived timing tightness (not true latency)
**Files:** `neuralnote_impl.cpp`

All MIDI events (provisional and CNN) are currently stamped at frame `0` (start of
the buffer). The downstream synth starts the note at sample 0 of its output buffer
regardless of where in the input buffer the onset actually occurred.

Fix: track which sample in the current block triggered the onset and use that as the
MIDI frame offset:

```cpp
// In the onset detection block:
uint32_t onsetFrame = 0; // current: always 0
// ... after onset fires:
onsetFrame = static_cast<uint32_t>(
    static_cast<float>(nSamples) * (1.0f - blockRms / peakRms)); // approximation
// Pass onsetFrame to writeMidi() instead of 0
```

**Trade-off:** approximation only; true sample-accurate onset requires per-sample RMS.

---

### 6. Per-range parallel worker threads
**Effort:** moderate — refactor worker thread per RangeState
**Gain:** inference time = max(range times) instead of sum; ~4× on Pi5 with 5 ranges
**Files:** `neuralnote_impl.cpp`, `neuralnote_tune.cpp`, `NeuralNoteShared.h`

Currently all ranges run sequentially in one shared worker thread. With 5 ranges and
~95 ms inference each, worst-case CNN latency for the last range is ~475 ms.
One thread per range, each blocking on its own semaphore, reduces this to ~95 ms.

**Implementation sketch:**
- Move `workerThread`, `workerQuit`, `workerSem` into `RangeStateBase`.
- Each range's thread wakes on its own semaphore (already one `sem_t` per
  `SnapshotChannel` — repurpose or add a second).
- The shared worker sem in `NeuralNotePlugin` becomes a count-down: audio thread
  posts to each range's sem individually.
- `modeVal` / `ampFloorVal` atomics remain on the plugin; worker reads them per inference.

**Trade-off:** 5 threads on a 4-core CPU — minor; inference is compute-bound so
threads won't truly parallelize past 4, but 5×95 ms → max(95 ms) is still a big win.

---

## Recommended Implementation Order

| Priority | Item | Effort | Gain |
|---|---|---|---|
| 1 | **MPM fallback** (item 1) | Small | Eliminates suppressed provisionals |
| 2 | **Double-drain midiOut** (item 2) | Trivial | ~1 JACK buffer CNN latency |
| 3 | **JACK graph order** (item 3) | Config | ~1 JACK buffer if mis-ordered |
| 4 | **JACK buffer size** (item 4) | Config | 3–8 ms |
| 5 | **MIDI frame timestamp** (item 5) | Small | Timing precision |
| 6 | **Parallel per-range workers** (item 6) | Moderate | ~4× CNN throughput |

---

## Already Implemented

- **Onset-triggered early dispatch** — `dispatchSnapshotIfReady` with `onsetFired` flag
  forces dispatch immediately on onset regardless of `freshSamples` threshold.
- **25 ms minimum fresh floor** — `MIN_FRESH_FLOOR` caps the dispatch wait at 25 ms.
- **Per-range window tuning** — `neuralnote_ranges.conf` / `neuralnote_tune.conf`.
- **Two-phase OBP + MPM + CNN** — provisional fires in ~5–15 ms; CNN confirms/corrects.
- **CNN staleness check** — stale CNN results don't cancel newer provisionals.
