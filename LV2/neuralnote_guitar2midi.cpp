/**
 * PiPitch — LV2 CPU-dispatch wrapper
 *
 * This is the only .so the LV2 host loads (referenced in manifest.ttl).
 * It contains no audio-processing code; instead it:
 *
 *   1. Reads Linux AT_HWCAP (aarch64) or uses __builtin_cpu_supports (x86-64)
 *      to detect the CPU's SIMD capabilities at runtime.
 *   2. dlopen()s the appropriate pre-compiled implementation .so from the
 *      same bundle directory:
 *        aarch64:
 *          dotprod + fp16 → neuralnote_impl_armv82.so   (Raspberry Pi 5)
 *          NEON only      → neuralnote_impl_neon.so     (Raspberry Pi 4)
 *        x86-64:
 *          AVX2 + FMA     → neuralnote_impl_avx2.so
 *          SSE4.2         → neuralnote_impl_sse42.so
 *   3. Delegates every LV2 callback to the loaded implementation via
 *      thin trampolines (no overhead after the first instantiate() call).
 *
 * Bundle layout:
 *   neuralnote_guitar2midi.lv2/
 *     neuralnote_guitar2midi.so    ← this file (host entry point)
 *     neuralnote_impl_neon.so      ← Pi 4 / generic aarch64
 *     neuralnote_impl_armv82.so    ← Pi 5 / Cortex-A76
 *     neuralnote_impl_sse42.so     ← x86-64 baseline
 *     neuralnote_impl_avx2.so      ← x86-64 modern
 *     manifest.ttl
 *     plugin.ttl
 *     ModelData/
 */

#include <lv2/core/lv2.h>

#include <dlfcn.h>
#include <cstring>
#include <mutex>
#include <string>

// ── CPU feature detection ────────────────────────────────────────────────────

#if defined(__aarch64__)
#  include <sys/auxv.h>
// Bit positions in AT_HWCAP for Linux aarch64 (see asm/hwcap.h)
#  ifndef HWCAP_ASIMDDP
#    define HWCAP_ASIMDDP (1UL << 20)   // integer dot-product (dotprod)
#  endif
#  ifndef HWCAP_ASIMDHP
#    define HWCAP_ASIMDHP (1UL << 10)   // half-precision SIMD (fp16)
#  endif

static const char* select_impl()
{
    const unsigned long hwcap = getauxval(AT_HWCAP);
    const bool has_dotprod  = (hwcap & HWCAP_ASIMDDP) != 0;
    const bool has_asimd_fp16 = (hwcap & HWCAP_ASIMDHP) != 0;
    // Require both dotprod and fp16: both are part of ARMv8.2-A and present
    // on Cortex-A76 (Pi 5) but absent on Cortex-A72 (Pi 4).
    return (has_dotprod && has_asimd_fp16) ? "neuralnote_impl_armv82.so"
                                           : "neuralnote_impl_neon.so";
}

#elif defined(__x86_64__) || defined(_M_X64)
static const char* select_impl()
{
    return __builtin_cpu_supports("avx2") ? "neuralnote_impl_avx2.so"
                                          : "neuralnote_impl_sse42.so";
}

#else
static const char* select_impl()
{
    return "neuralnote_impl_baseline.so";
}
#endif

// ── Runtime state ────────────────────────────────────────────────────────────

using ImplEntryFn = const LV2_Descriptor* (*)(uint32_t);

static std::once_flag          g_load_once;
static void*                   g_impl_handle = nullptr;
static const LV2_Descriptor*   g_impl_desc   = nullptr;

static void load_impl_once(const char* bundle_path)
{
    const char* impl_name = select_impl();

    std::string dir = bundle_path ? bundle_path : "";
    if (!dir.empty() && dir.back() != '/') dir += '/';
    const std::string path = dir + impl_name;

    g_impl_handle = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (!g_impl_handle) {
        // Cannot log here — logger is not yet initialised.
        // The host will see nullptr from lv2_descriptor and report the error.
        return;
    }

    const auto entry = reinterpret_cast<ImplEntryFn>(
        dlsym(g_impl_handle, "neuralnote_impl_descriptor"));
    if (!entry) {
        dlclose(g_impl_handle);
        g_impl_handle = nullptr;
        return;
    }

    g_impl_desc = entry(0);
    if (!g_impl_desc) {
        dlclose(g_impl_handle);
        g_impl_handle = nullptr;
    }
}

// ── Trampoline callbacks ──────────────────────────────────────────────────────

static LV2_Handle w_instantiate(const LV2_Descriptor* /*desc*/,
                                 double                rate,
                                 const char*           bundle_path,
                                 const LV2_Feature* const* features)
{
    // Load the impl .so exactly once, using the bundle path from this call.
    std::call_once(g_load_once, load_impl_once, bundle_path);
    if (!g_impl_desc) return nullptr;
    return g_impl_desc->instantiate(g_impl_desc, rate, bundle_path, features);
}

static void w_connect_port(LV2_Handle h, uint32_t port, void* data)
{
    g_impl_desc->connect_port(h, port, data);
}

static void w_activate(LV2_Handle h)
{
    if (g_impl_desc->activate) g_impl_desc->activate(h);
}

static void w_run(LV2_Handle h, uint32_t n_samples)
{
    g_impl_desc->run(h, n_samples);
}

static void w_deactivate(LV2_Handle h)
{
    if (g_impl_desc->deactivate) g_impl_desc->deactivate(h);
}

static void w_cleanup(LV2_Handle h)
{
    g_impl_desc->cleanup(h);
}

static const void* w_extension_data(const char* uri)
{
    if (g_impl_desc && g_impl_desc->extension_data)
        return g_impl_desc->extension_data(uri);
    return nullptr;
}

// ── LV2 descriptor ───────────────────────────────────────────────────────────

static const LV2_Descriptor wrapper_descriptor = {
    "https://github.com/anirbanray1981/PiPitch",
    w_instantiate,
    w_connect_port,
    w_activate,
    w_run,
    w_deactivate,
    w_cleanup,
    w_extension_data,
};

LV2_SYMBOL_EXPORT const LV2_Descriptor* lv2_descriptor(uint32_t index)
{
    return (index == 0) ? &wrapper_descriptor : nullptr;
}
