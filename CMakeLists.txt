cmake_minimum_required(VERSION 3.16)
project(PiPitch VERSION 1.0.0)

# C++ settings
set(CMAKE_CXX_STANDARD 17)
enable_language(CXX)
set(CMAKE_EXPORT_COMPILE_COMMANDS TRUE)
set(CMAKE_POSITION_INDEPENDENT_CODE TRUE)

option(LTO "Enable Link Time Optimization" ON)

# ── NeuralNote submodule ─────────────────────────────────────────────────────
# RTNeural is needed for BasicPitch CNN inference.
# The NeuralNote submodule's own CMakeLists.txt adds JUCE targets which we
# don't need, so we only pull in RTNeural here.
set(NN "${CMAKE_SOURCE_DIR}/NeuralNote")
add_subdirectory(${NN}/ThirdParty/RTNeural)

# ── LV2 Plugin target ────────────────────────────────────────────────────────
option(BUILD_LV2 "Build LV2 plugin" OFF)

# Inference window fed to BasicPitch (ms). Smaller = lower latency.
set(LV2_WINDOW_MS 500 CACHE STRING "Inference window length in ms (default 500)")

if(BUILD_LV2)
    set(LV2_BUNDLE_NAME "neuralnote_guitar2midi.lv2")
    set(LV2_BUNDLE_DIR  "${CMAKE_BINARY_DIR}/${LV2_BUNDLE_NAME}")

    # Collect NeuralNote's Lib/Model sources
    file(GLOB_RECURSE LIB_MODEL_SOURCES
        "${NN}/Lib/Model/*.cpp"
    )

    # ── Shared dependency discovery ───────────────────────────────────────────
    find_package(PkgConfig REQUIRED)
    pkg_check_modules(LV2 REQUIRED lv2)

    find_library(LV2_ONNXRUNTIME_LIB
        NAMES onnxruntime
        PATHS
            "${NN}/ThirdParty/onnxruntime/lib-linux-aarch64"
            /usr/local/lib
        NO_DEFAULT_PATH
    )
    if(NOT LV2_ONNXRUNTIME_LIB)
        find_library(LV2_ONNXRUNTIME_LIB onnxruntime REQUIRED)
    endif()
    message(STATUS "LV2: using onnxruntime: ${LV2_ONNXRUNTIME_LIB}")

    # ── Helper: create one implementation .so ─────────────────────────────────
    function(neuralnote_add_impl SUFFIX OUTPUT_NAME)
        set(MARCH_FLAGS ${ARGN})
        add_library(NeuralNoteImpl_${SUFFIX} MODULE
            ${CMAKE_SOURCE_DIR}/LV2/neuralnote_impl.cpp
            ${LIB_MODEL_SOURCES}
        )
        set_target_properties(NeuralNoteImpl_${SUFFIX} PROPERTIES
            PREFIX ""
            OUTPUT_NAME "${OUTPUT_NAME}"
            LIBRARY_OUTPUT_DIRECTORY "${LV2_BUNDLE_DIR}"
        )
        target_compile_features(NeuralNoteImpl_${SUFFIX} PRIVATE cxx_std_17)
        target_compile_options(NeuralNoteImpl_${SUFFIX} PRIVATE
            -O3 -ffast-math -fPIC ${MARCH_FLAGS}
        )
        target_include_directories(NeuralNoteImpl_${SUFFIX} PRIVATE
            ${CMAKE_SOURCE_DIR}/LV2
            ${NN}/Lib/Model
            ${NN}/Lib/Utils
            ${NN}/ThirdParty/RTNeural
            ${NN}/ThirdParty/onnxruntime/include
            ${LV2_INCLUDE_DIRS}
        )
        target_compile_definitions(NeuralNoteImpl_${SUFFIX} PRIVATE
            SAVE_DOWNSAMPLED_AUDIO=0
            USE_TEST_NOTE_FRAME_TO_TIME=0
            "NEURALNOTE_IMPL_NAME=\"${OUTPUT_NAME}\""
            "LV2_WINDOW_MS=${LV2_WINDOW_MS}"
        )
        target_link_libraries(NeuralNoteImpl_${SUFFIX} PRIVATE
            RTNeural
            ${LV2_ONNXRUNTIME_LIB}
            ${LV2_LIBRARIES}
            pthread dl
        )
    endfunction()

    # ── Implementation .so files (one per SIMD tier) ──────────────────────────
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
        neuralnote_add_impl(NEON   "neuralnote_impl_neon"   -march=armv8-a+simd)
        neuralnote_add_impl(ARMv82 "neuralnote_impl_armv82" -march=armv8.2-a+dotprod+fp16)
        message(STATUS "LV2: building aarch64 impls: neon + armv8.2-a+dotprod+fp16")
    elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
        neuralnote_add_impl(SSE42  "neuralnote_impl_sse42"  -msse4.2)
        neuralnote_add_impl(AVX2   "neuralnote_impl_avx2"   -mavx2 -mfma)
        message(STATUS "LV2: building x86-64 impls: sse4.2 + avx2+fma")
    else()
        neuralnote_add_impl(Baseline "neuralnote_impl_baseline")
        message(STATUS "LV2: building generic baseline impl")
    endif()

    # ── Wrapper .so (the only target the LV2 host loads) ─────────────────────
    add_library(NeuralNoteGuitar2Midi_LV2 MODULE
        ${CMAKE_SOURCE_DIR}/LV2/neuralnote_guitar2midi.cpp
    )
    set_target_properties(NeuralNoteGuitar2Midi_LV2 PROPERTIES
        PREFIX ""
        OUTPUT_NAME "neuralnote_guitar2midi"
        LIBRARY_OUTPUT_DIRECTORY "${LV2_BUNDLE_DIR}"
    )
    target_compile_features(NeuralNoteGuitar2Midi_LV2 PRIVATE cxx_std_17)
    target_compile_options(NeuralNoteGuitar2Midi_LV2 PRIVATE -O2 -fPIC)
    target_include_directories(NeuralNoteGuitar2Midi_LV2 PRIVATE
        ${LV2_INCLUDE_DIRS}
    )
    target_link_libraries(NeuralNoteGuitar2Midi_LV2 PRIVATE
        ${LV2_LIBRARIES}
        dl
    )

    # ── Bundle assembly ───────────────────────────────────────────────────────
    add_custom_command(TARGET NeuralNoteGuitar2Midi_LV2 POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy
            ${CMAKE_SOURCE_DIR}/LV2/manifest.ttl
            ${LV2_BUNDLE_DIR}/manifest.ttl
        COMMAND ${CMAKE_COMMAND} -E copy
            ${CMAKE_SOURCE_DIR}/LV2/plugin.ttl
            ${LV2_BUNDLE_DIR}/plugin.ttl
        COMMAND ${CMAKE_COMMAND} -E copy
            ${CMAKE_SOURCE_DIR}/LV2/neuralnote_ranges.conf
            ${LV2_BUNDLE_DIR}/neuralnote_ranges.conf
        COMMAND ${CMAKE_COMMAND} -E copy_directory
            ${NN}/Lib/ModelData
            ${LV2_BUNDLE_DIR}/ModelData
        COMMAND ${CMAKE_COMMAND} -E copy
            ${CMAKE_SOURCE_DIR}/LV2/swift_f0_model.onnx
            ${LV2_BUNDLE_DIR}/swift_f0_model.onnx
        COMMENT "Assembling LV2 bundle: ${LV2_BUNDLE_DIR}"
    )

    message(STATUS "LV2 bundle will be assembled at: ${LV2_BUNDLE_DIR}")

    # ── Latency benchmark ─────────────────────────────────────────────────────
    option(BUILD_LV2_BENCH "Build the Guitar2MIDI latency benchmark" ON)
    if(BUILD_LV2_BENCH)
        add_executable(latency_bench
            ${CMAKE_SOURCE_DIR}/LV2/latency_bench.cpp
            ${LIB_MODEL_SOURCES}
        )
        target_compile_features(latency_bench PRIVATE cxx_std_17)
        target_compile_options(latency_bench PRIVATE -O2 -fPIC)
        target_include_directories(latency_bench PRIVATE
            ${CMAKE_SOURCE_DIR}/LV2
            ${NN}/Lib/Model
            ${NN}/Lib/Utils
            ${NN}/ThirdParty/RTNeural
            ${NN}/ThirdParty/onnxruntime/include
        )
        target_compile_definitions(latency_bench PRIVATE
            SAVE_DOWNSAMPLED_AUDIO=0
            USE_TEST_NOTE_FRAME_TO_TIME=0
            "LV2_WINDOW_MS=${LV2_WINDOW_MS}"
        )
        target_link_libraries(latency_bench PRIVATE
            RTNeural
            ${LV2_ONNXRUNTIME_LIB}
            pthread dl
        )
        message(STATUS "LV2: latency_bench target enabled")
    endif()

    option(BUILD_LV2_TUNE "Build the parameter sweep / latency tuning tool (requires JACK)" ON)
    if(BUILD_LV2_TUNE)
        if(NOT JACK_LIB)
            find_library(JACK_LIB jack REQUIRED)
        endif()
        add_executable(neuralnote_tune
            ${CMAKE_SOURCE_DIR}/LV2/neuralnote_tune.cpp
            ${LIB_MODEL_SOURCES}
        )
        target_compile_features(neuralnote_tune PRIVATE cxx_std_17)
        target_compile_options(neuralnote_tune PRIVATE -O2 -fPIC)
        target_include_directories(neuralnote_tune PRIVATE
            ${CMAKE_SOURCE_DIR}/LV2
            ${NN}/Lib/Model
            ${NN}/Lib/Utils
            ${NN}/ThirdParty/RTNeural
            ${NN}/ThirdParty/onnxruntime/include
        )
        target_compile_definitions(neuralnote_tune PRIVATE
            SAVE_DOWNSAMPLED_AUDIO=0
            USE_TEST_NOTE_FRAME_TO_TIME=0
        )
        find_library(FFTW3F_LIB fftw3f)
        target_link_libraries(neuralnote_tune PRIVATE
            RTNeural
            ${LV2_ONNXRUNTIME_LIB}
            ${JACK_LIB}
            $<$<BOOL:${FFTW3F_LIB}>:${FFTW3F_LIB}>
            pthread dl
        )
        message(STATUS "LV2: neuralnote_tune target enabled")
    endif()

    # ── Install targets ───────────────────────────────────────────────────────
    set(LV2_INSTALL_PATH "/zynthian/zynthian-plugins/lv2"
        CACHE STRING "Destination for the LV2 bundle (default: Zynthian LV2 path)")

    install(DIRECTORY "${LV2_BUNDLE_DIR}/"
        DESTINATION "${LV2_INSTALL_PATH}/${LV2_BUNDLE_NAME}"
        USE_SOURCE_PERMISSIONS
    )

    install(PROGRAMS "${CMAKE_SOURCE_DIR}/LV2/neuralnote-connect.sh"
        DESTINATION /usr/local/bin
        RENAME neuralnote-connect.sh
    )

    if(BUILD_LV2_TUNE)
        install(TARGETS neuralnote_tune
            RUNTIME DESTINATION /usr/local/bin
        )
    endif()

    install(FILES "${CMAKE_SOURCE_DIR}/LV2/neuralnote-connect.service"
        DESTINATION /etc/systemd/system
    )

    install(CODE "
        message(STATUS \"-- Running systemctl daemon-reload\")
        execute_process(COMMAND systemctl daemon-reload)

        execute_process(
            COMMAND systemctl is-enabled neuralnote-connect.service
            OUTPUT_VARIABLE _svc_enabled OUTPUT_STRIP_TRAILING_WHITESPACE
            RESULT_VARIABLE _svc_rc
        )
        if(NOT _svc_enabled STREQUAL \"enabled\")
            message(STATUS \"-- Enabling neuralnote-connect.service\")
            execute_process(COMMAND systemctl enable neuralnote-connect.service)
        else()
            message(STATUS \"-- neuralnote-connect.service already enabled\")
        endif()

        message(STATUS \"-- Restarting zynthian.service\")
        execute_process(COMMAND systemctl restart zynthian.service)

        message(STATUS \"-- Starting neuralnote-connect.service\")
        execute_process(COMMAND systemctl restart neuralnote-connect.service)
    ")

endif()
