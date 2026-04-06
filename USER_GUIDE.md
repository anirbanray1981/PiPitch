# PiPitch User Guide

Real-time guitar-to-MIDI LV2 plugin for Raspberry Pi 5 (Zynthian).
Plug in your guitar, load the plugin, and play synths with MIDI.

---

## Quick Start

1. Load **PiPitch** as an LV2 effect in your host (Zynthian, Ardour, Carla, etc.)
2. Route your guitar audio into `audio_in`
3. Route `midi_out` to a synth
4. Play -- the default mode (GoertzelMono) works out of the box

The plugin passes audio through on `audio_out` so you can monitor your dry signal.

---

## Modes

PiPitch has 6 detection modes. Choose based on your playing style:

| Mode | # | Best for | Latency |
|------|---|----------|---------|
| **GoertzelMono** | 4 | Single notes, fastest response | 15--60 ms |
| **GoertzelPoly** | 5 | Chords (up to 6 notes) | 15--60 ms + Neural Network confirmation |
| **SwiftMono** | 2 | Single notes with pitch bend | 70--140 ms |
| **SwiftPoly** | 3 | Fast note-on + polyphonic sustain | ~100 ms |
| **Mono** | 1 | Single notes detected by Neural Network | 95--185 ms |
| **Poly** | 0 | Polyphonic detected by Neural Network | 95--185 ms |

**Recommended:** Start with **GoertzelMono** (default). Switch to **GoertzelPoly** for chords or **SwiftMono** if you want pitch bend for vibrato/bends.

---

## Controls

| Control | Default | What it does |
|---------|---------|--------------|
| **Mode** | GoertzelMono (4) | Detection algorithm (see above) |
| **Onset sensitivity** | 0.6 | How easily a new note triggers. Lower = more sensitive. |
| **Noise gate floor** | 0.003 | Below this level, input is treated as silence. Adjust to compensate the noise from your guitar input. |
| **Amplitude floor** | 0.3 | Neural Network confidence floor (poly/mono/goertzelpoly modes). Lower catches quieter notes but may add false detections. |
| **Frame threshold** | 0.4 | Per-frame Neural Network confidence (poly/mono/goertzelpoly). Lower = more permissive. |
| **Onset blank (ms)** | 25 | Minimum time between re-triggers. Raise if you get stuttering on sustained notes. |
| **Provisional** | On (0) | Fast note detection before Neural Network confirms. "On" for lowest latency; "None" for safest. |
| **Pitch bend** | Off | 14-bit pitch bend tracking (SwiftMono only). Enable for vibrato and string bends. |
| **Max polyphony** | 3 | Maximum simultaneous notes in GoertzelPoly mode (1--6). |

---

## Tuning Tips

### Getting clean single-note tracking
- Use **GoertzelMono** mode
- If ghost notes appear on string decay, raise **noise gate floor** slightly (try 0.005--0.01)
- If notes re-trigger on sustained playing, increase **onset blank** to 30--50 ms

### Playing chords
- Use **GoertzelPoly** mode
- Set **max polyphony** to match your chord voicings (3 for triads, 4--6 for fuller chords)
- Keep **amplitude floor** at 0.3 and **frame threshold** at 0.4 -- higher values miss quieter strings

### Using pitch bend (vibrato/bends)
- Use **SwiftMono** mode with **pitch bend = On**
- Bend range is +/- 2 semitones
- Works best on sustained notes after the attack transient settles

### Reducing latency
- **GoertzelMono** is the fastest mode (15--60 ms depending on pitch)
- Higher notes are detected faster than lower notes in all modes
- **Provisional = On** fires a fast note guess before the Neural Network confirms (poly/mono modes)

### Reducing false notes
- Raise **noise gate floor** if your signal is noisy
- Set **provisional = None** to only emit Neural Network-confirmed notes (higher latency but no wrong guesses)
- In **GoertzelPoly**, the Neural Network automatically vetoes harmonic ghosts from the Goertzel scout

### Recommended Guitar Level
- The plugin works best at -12 to -18 dbfs input level from the guitar.

---

## MIDI Routing (Zynthian)

PiPitch includes an auto-connect script that wires MIDI to all synth chains at boot.

**GoertzelMono/GoertzelPoly** output on **MIDI channel 2**. Set your synth chain to listen on channel 2.

All other modes output on **MIDI channel 1** (default).

To send MIDI to multiple synth chains simultaneously, set in `/zynthian/config/midi-profiles/default.sh`:
```
export ZYNTHIAN_MIDI_SINGLE_ACTIVE_CHANNEL="0"
```

---

## Note Range

PiPitch detects notes from **E2 (MIDI 40)** to **E6 (MIDI 88)** -- the standard guitar range plus harmonics. Notes outside this range are ignored.

The detection range is split into 5 internal zones, each tuned for its frequency range (longer analysis windows for bass notes, shorter for treble). This is configured in `pipitch_ranges.conf` and generally doesn't need adjustment.

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| No MIDI output | Check audio routing -- guitar must be connected to `audio_in` |
| Ghost notes from silence | Raise **noise gate floor** (try 0.01) |
| Wrong octave detected | Normal for Neural Network modes on low notes; use **GoertzelMono** for best accuracy |
| Notes stutter/re-trigger | Raise **onset blank** to 40--50 ms |
| Chords missing notes | Lower **amplitude floor** to 0.2; use **GoertzelPoly** mode |
| High latency | Switch to **GoertzelMono**; ensure you're on Pi 5 |
| Pitch bend not working | Only available in **SwiftMono** mode with **bend = On** |
