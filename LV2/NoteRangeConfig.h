#pragma once

/**
 * NoteRangeConfig — per-note-range parameter configuration
 *
 * Config file format (INI-style, default path: pipitch_tune.conf next to binary,
 * or pipitch_ranges.conf inside the LV2 bundle):
 *
 *   # Global settings (pipitch_tune.conf only — not used by the LV2 plugin conf)
 *   gate_floor      = 0.003
 *   amp_floor       = 0.65
 *   threshold       = 0.6
 *   frame_threshold = 0.5
 *   mode            = poly    # or mono, swiftmono, swiftpoly
 *   onset_blank_ms  = 25      # re-trigger suppression window (ms)
 *   swift_threshold = 0.5     # SwiftF0 per-frame confidence threshold (swiftmono mode)
 *   provisional    = on      # on | swift | none — controls OBP+MPM provisional detection
 *
 *   [range]
 *   name            = E2-B2
 *   midi_low        = 40
 *   midi_high       = 47
 *   window          = 120
 *   min_note_length = 6
 *   hold_cycles     = 4
 *   swift_hold_cycles = 2
 *
 *   [range]
 *   name            = C3-B3
 *   midi_low        = 48
 *   midi_high       = 59
 *   window          = 80
 *   min_note_length = 4
 *   hold_cycles     = 2
 *
 * Ranges should be non-overlapping and listed in ascending MIDI order.
 * Notes outside all defined ranges are silently discarded.
 */

#include <cstdio>
#include <string>
#include <vector>

enum class PlayMode { POLY, MONO, SWIFT_MONO, SWIFT_POLY };
enum class ProvMode { ON = 0, SWIFT = 1, NONE = 2 };

struct NoteRange {
    std::string name         = "default";
    int   midiLow            = 0;
    int   midiHigh           = 127;
    float windowMs           = 150.0f;
    int   minNoteLength      = 6;      // CNN frames
    int   holdCycles         = 2;      // inference cycles to hold OFF for this range
    int   swiftHoldCycles    = 2;      // hold cycles when using SwiftF0 (faster cycle time)
};

struct RangeConfig {
    std::vector<NoteRange> ranges;
    float     gateFloor      = 0.003f;
    float     ampFloor       = 0.65f;
    float     threshold      = 0.6f;   // onset sensitivity (0.05–0.95)
    float     frameThreshold = 0.5f;   // frame confidence (0.05–0.95)
    float     onsetBlankMs      = 25.0f;  // re-trigger suppression window (ms)
    float     swiftF0Threshold  = 0.5f;   // SwiftF0 per-frame confidence (swiftmono mode)
    float     octaveLockMs      = 250.0f;  // suppress ±12/24 semitone jumps in swiftmono (0=off)
    PlayMode  mode              = PlayMode::MONO;
    ProvMode  provisionalMode  = ProvMode::ON;
    bool      bendEnabled      = false;  // 14-bit MIDI pitch bend (swiftmono only)
};

// Return the first range whose [midiLow, midiHigh] contains pitch, or nullptr.
static inline const NoteRange* findNoteRange(const RangeConfig& cfg, int pitch)
{
    for (const auto& r : cfg.ranges)
        if (pitch >= r.midiLow && pitch <= r.midiHigh)
            return &r;
    return nullptr;
}

// Parse a simple INI-style config file.
// Returns an empty RangeConfig (ranges.empty()) if the file cannot be opened.
static inline RangeConfig loadRangeConfig(const std::string& path)
{
    RangeConfig cfg;

    std::FILE* f = std::fopen(path.c_str(), "r");
    if (!f) return cfg;

    auto trim = [](const char* s) -> std::string {
        std::string t(s);
        size_t a = t.find_first_not_of(" \t\r\n");
        size_t b = t.find_last_not_of(" \t\r\n");
        return (a == std::string::npos) ? "" : t.substr(a, b - a + 1);
    };

    char line[512];
    NoteRange* cur = nullptr;

    while (std::fgets(line, sizeof(line), f)) {
        std::string s = trim(line);
        if (s.empty() || s[0] == '#' || s[0] == ';') continue;

        if (s == "[range]") {
            cfg.ranges.emplace_back();
            cur = &cfg.ranges.back();
            continue;
        }

        auto eq = s.find('=');
        if (eq == std::string::npos) continue;
        std::string key = trim(s.substr(0, eq).c_str());
        std::string val = trim(s.substr(eq + 1).c_str());
        if (val.empty()) continue;

        // Global keys — valid anywhere in the file
        if (key == "gate_floor")      { cfg.gateFloor      = std::stof(val); continue; }
        if (key == "amp_floor")       { cfg.ampFloor       = std::stof(val); continue; }
        if (key == "threshold")       { cfg.threshold      = std::stof(val); continue; }
        if (key == "frame_threshold") { cfg.frameThreshold = std::stof(val); continue; }
        if (key == "onset_blank_ms")  { cfg.onsetBlankMs     = std::stof(val); continue; }
        if (key == "swift_threshold") { cfg.swiftF0Threshold = std::stof(val); continue; }
        if (key == "mode") {
            if      (val == "mono")      cfg.mode = PlayMode::MONO;
            else if (val == "swiftmono") cfg.mode = PlayMode::SWIFT_MONO;
            else if (val == "swiftpoly") cfg.mode = PlayMode::SWIFT_POLY;
            else                         cfg.mode = PlayMode::POLY;
            continue;
        }
        if (key == "octave_lock_ms")  { cfg.octaveLockMs    = std::stof(val); continue; }
        if (key == "bend") {
            cfg.bendEnabled = (val == "on" || val == "true" || val == "1");
            continue;
        }
        if (key == "provisional") {
            if      (val == "swift") cfg.provisionalMode = ProvMode::SWIFT;
            else if (val == "none" || val == "off") cfg.provisionalMode = ProvMode::NONE;
            else                     cfg.provisionalMode = ProvMode::ON;
            continue;
        }

        if (!cur) continue; // range-specific keys only valid inside [range]

        if      (key == "name")            cur->name          = val;
        else if (key == "midi_low")        cur->midiLow       = std::stoi(val);
        else if (key == "midi_high")       cur->midiHigh      = std::stoi(val);
        else if (key == "window")          cur->windowMs      = std::stof(val);
        else if (key == "min_note_length") cur->minNoteLength = std::stoi(val);
        else if (key == "hold_cycles")       cur->holdCycles      = std::stoi(val);
        else if (key == "swift_hold_cycles") cur->swiftHoldCycles = std::stoi(val);
    }

    std::fclose(f);
    return cfg;
}
