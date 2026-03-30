#!/bin/bash
# pipitch-connect.sh — MIDI fan-out for PiPitch
#
# Connects ZynMidiRouter:ch0_out (PiPitch's MIDI output channel) to
# additional synth chains that Zynthian does not automatically route there.
#
# Runs at boot via pipitch-connect.service (After=zynthian.service).
# Edit the EXTRA_DSTS array to match your Zynthian synth chain layout.
# ──────────────────────────────────────────────────────────────────────────

# PiPitch MIDI output → ZynMidiRouter input (essential — always needed)
PIPITCH_SRC="PiPitch-01:midi_out"
PIPITCH_DST="ZynMidiRouter:dev0_in"

# ZynMidiRouter fan-out to additional synth chains.
# zynaddsubfx-01 is managed by Zynthian's autoconnect automatically;
# list only the chains it does NOT connect on its own.
FANOUT_SRC="ZynMidiRouter:ch0_out"
FANOUT_DSTS=(
    "fluidsynth:midi_00"
    "LinuxSampler:midi_in_0"
)

MAX_WAIT=60   # seconds to wait for each port before giving up
POLL=2        # polling interval in seconds

wait_for_port() {
    local port="$1"
    local elapsed=0
    while [ "$elapsed" -lt "$MAX_WAIT" ]; do
        jack_lsp 2>/dev/null | grep -qF "$port" && return 0
        sleep "$POLL"
        elapsed=$(( elapsed + POLL ))
    done
    echo "pipitch-connect: timeout waiting for port '$port'" >&2
    return 1
}

connect() {
    local src="$1" dst="$2"
    wait_for_port "$src" || return 1
    wait_for_port "$dst" || return 1
    # No-op if already connected
    jack_lsp -c "$src" 2>/dev/null | grep -qF "$dst" && return 0
    if jack_connect "$src" "$dst" 2>/dev/null; then
        echo "pipitch-connect: connected  $src  ->  $dst"
    else
        echo "pipitch-connect: failed to connect  $src  ->  $dst" >&2
    fi
}

# Primary: PiPitch → ZynMidiRouter
connect "$PIPITCH_SRC" "$PIPITCH_DST"

# Fan-out: ZynMidiRouter → extra synth chains
for dst in "${FANOUT_DSTS[@]}"; do
    connect "$FANOUT_SRC" "$dst"
done
