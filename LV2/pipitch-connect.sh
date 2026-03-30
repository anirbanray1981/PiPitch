#!/bin/bash
# pipitch-connect.sh — MIDI routing for PiPitch LV2 plugin
#
# 1. Waits for the PiPitch JACK port to appear
# 2. Connects PiPitch:midi_out → ZynMidiRouter (for Zynthian integration)
# 3. Discovers all synth engine MIDI inputs and connects PiPitch directly
#    to each one (bypasses ZynMidiRouter for lowest latency)
#
# Runs at boot via pipitch-connect.service (After=zynthian.service).
# ──────────────────────────────────────────────────────────────────────────

PIPITCH_PORT="PiPitch-01:midi_out"
ZYNROUTER_IN="ZynMidiRouter:dev0_in"

MAX_WAIT=60   # seconds to wait for PiPitch port
POLL=2        # polling interval

# ── Helpers ───────────────────────────────────────────────────────────────

wait_for_port() {
    local port="$1" elapsed=0
    while [ "$elapsed" -lt "$MAX_WAIT" ]; do
        jack_lsp 2>/dev/null | grep -qF "$port" && return 0
        sleep "$POLL"
        elapsed=$(( elapsed + POLL ))
    done
    echo "pipitch-connect: timeout waiting for '$port'" >&2
    return 1
}

connect() {
    local src="$1" dst="$2"
    # Skip if already connected
    jack_lsp -c "$src" 2>/dev/null | grep -qF "$dst" && return 0
    if jack_connect "$src" "$dst" 2>/dev/null; then
        echo "pipitch-connect: connected  $src  →  $dst"
    else
        echo "pipitch-connect: FAILED     $src  →  $dst" >&2
    fi
}

discover_synth_midi_inputs() {
    # Find MIDI input ports belonging to synth engines.
    # Checks each port with jack_port_info: must be MIDI type + input direction.
    # Filters out infrastructure (router, sequencer, system, PiPitch itself).
    local port props ptype
    while IFS= read -r port; do
        # Read the two metadata lines
        read -r props
        read -r ptype

        # Must be MIDI type and input direction
        [[ "$ptype" == *midi* ]] || continue
        [[ "$props" == *input* ]] || continue

        # Skip infrastructure and non-synth ports
        case "$port" in
            PiPitch*|pipitch*) continue ;;
            ZynMidiRouter*|ZynMaster*) continue ;;
            system:midi*|system:capture*|system:playback*) continue ;;
            ttymidi:*|a2j:*) continue ;;
            zynsmf:*|zynseq:*|zynmixer:*) continue ;;
            audioplayer:*) continue ;;
            *"Midi Through"*) continue ;;
            # Skip audio ports that snuck through
            *out-L|*out-R|*out_*|*output*|*input_[0-9]*) continue ;;
        esac

        echo "$port"
    done < <(jack_lsp -tp 2>/dev/null)
}

# ── Main ──────────────────────────────────────────────────────────────────

echo "pipitch-connect: waiting for PiPitch JACK port..."
wait_for_port "$PIPITCH_PORT" || exit 1

# Always connect to ZynMidiRouter (Zynthian needs this for its own routing)
connect "$PIPITCH_PORT" "$ZYNROUTER_IN"

# Discover and connect directly to all synth engine MIDI inputs (low latency)
echo "pipitch-connect: discovering synth engines..."
found=0
while IFS= read -r dst; do
    [ -z "$dst" ] && continue
    connect "$PIPITCH_PORT" "$dst"
    found=1
done < <(discover_synth_midi_inputs)

[ "$found" -eq 0 ] && echo "pipitch-connect: no synth MIDI inputs found"

echo "pipitch-connect: done"
