#!/usr/bin/env python3
"""Generate synthetic guitar-like WAV files for latency testing."""

import struct
import math
import sys
import os

SAMPLE_RATE = 48000
DURATION_S = 2.0  # 0.5s silence + 1.0s note + 0.5s silence
SILENCE_BEFORE = 0.5
NOTE_DURATION = 1.0

def midi_to_freq(midi):
    return 440.0 * (2.0 ** ((midi - 69) / 12.0))

def generate_guitar_note(freq, sr, duration, num_harmonics=12):
    """Generate a guitar-like waveform with harmonics and decay."""
    n_samples = int(sr * duration)
    samples = [0.0] * n_samples

    # Guitar harmonic amplitudes (roughly 1/n with some variation)
    harm_amps = [1.0]
    for h in range(1, num_harmonics):
        harm_amps.append(1.0 / ((h + 1) ** 0.8) * (0.8 + 0.4 * ((h % 3) == 0)))

    # Limit harmonics to Nyquist
    max_harm = num_harmonics
    for h in range(num_harmonics):
        if freq * (h + 1) >= sr / 2:
            max_harm = h
            break

    # Decay time constant (lower notes ring longer)
    decay_tau = 0.3 + 0.5 * (1.0 - min(freq, 1000.0) / 1000.0)

    # Attack: fast rise (~2ms) with pick transient
    attack_samples = int(sr * 0.002)
    pick_samples = int(sr * 0.001)  # 1ms broadband pick noise

    for i in range(n_samples):
        t = i / sr
        # Envelope: attack + decay
        if i < attack_samples:
            env = i / attack_samples
        else:
            env = math.exp(-(t - attack_samples / sr) / decay_tau)

        # Harmonic sum
        val = 0.0
        for h in range(max_harm):
            f_h = freq * (h + 1)
            # Each harmonic decays slightly faster
            h_decay = math.exp(-(t) / (decay_tau / (1.0 + h * 0.15)))
            val += harm_amps[h] * h_decay * math.sin(2.0 * math.pi * f_h * t)

        # Pick transient (broadband noise burst)
        if i < pick_samples:
            import random
            random.seed(i)  # deterministic
            val += 0.5 * (random.random() * 2 - 1) * (1.0 - i / pick_samples)

        samples[i] = val * env

    # Normalize to 0.8 peak
    peak = max(abs(s) for s in samples)
    if peak > 0:
        scale = 0.8 / peak
        samples = [s * scale for s in samples]

    return samples

def write_wav(path, samples, sr):
    """Write 32-bit float mono WAV."""
    n = len(samples)
    data_size = n * 4
    with open(path, 'wb') as f:
        # RIFF header
        f.write(b'RIFF')
        f.write(struct.pack('<I', 36 + data_size))
        f.write(b'WAVE')
        # fmt chunk (IEEE float)
        f.write(b'fmt ')
        f.write(struct.pack('<I', 16))       # chunk size
        f.write(struct.pack('<H', 3))        # IEEE float
        f.write(struct.pack('<H', 1))        # mono
        f.write(struct.pack('<I', sr))       # sample rate
        f.write(struct.pack('<I', sr * 4))   # byte rate
        f.write(struct.pack('<H', 4))        # block align
        f.write(struct.pack('<H', 32))       # bits per sample
        # data chunk
        f.write(b'data')
        f.write(struct.pack('<I', data_size))
        for s in samples:
            f.write(struct.pack('<f', s))

# Notes to generate
notes = {
    'E2': 40,
    'E3': 52,
    'E4': 64,
    'E5': 76,
    'E6': 88,
}

out_dir = sys.argv[1] if len(sys.argv) > 1 else '.'

for name, midi in notes.items():
    freq = midi_to_freq(midi)
    print(f"Generating {name} (MIDI {midi}, {freq:.1f} Hz)...")

    # Silence + note + silence
    silence_before = [0.0] * int(SAMPLE_RATE * SILENCE_BEFORE)
    note_samples = generate_guitar_note(freq, SAMPLE_RATE, NOTE_DURATION)
    silence_after = [0.0] * int(SAMPLE_RATE * 0.5)

    all_samples = silence_before + note_samples + silence_after

    path = os.path.join(out_dir, f'test_{name}.wav')
    write_wav(path, all_samples, SAMPLE_RATE)
    print(f"  -> {path} ({len(all_samples)} samples, {len(all_samples)/SAMPLE_RATE:.2f}s)")

    # Also create label file
    lbl_path = os.path.join(out_dir, f'test_{name}.lbl')
    with open(lbl_path, 'w') as f:
        f.write(f'mono\n{name}\n')
    print(f"  -> {lbl_path}")

print("\nDone.")
