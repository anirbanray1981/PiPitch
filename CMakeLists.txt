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
    set(LV2_BUNDLE_NAME "pipitch.lv2")
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

    # Resolve the directory containing libonnxruntime.so for rpath + bundle copy
    get_filename_component(LV2_ONNXRUNTIME_DIR "${LV2_ONNXRUNTIME_LIB}" DIRECTORY)
    get_filename_component(LV2_ONNXRUNTIME_NAME "${LV2_ONNXRUNTIME_LIB}" NAME)

    # ── Helper: create one implementation .so ─────────────────────────────────
    function(pipitch_add_impl SUFFIX OUTPUT_NAME)
        set(MARCH_FLAGS ${ARGN})
        add_library(PiPitchImpl_${SUFFIX} MODULE
            ${CMAKE_SOURCE_DIR}/LV2/pipitch_impl.cpp
            ${LIB_MODEL_SOURCES}
        )
        set_target_properties(PiPitchImpl_${SUFFIX} PROPERTIES
            PREFIX ""
            OUTPUT_NAME "${OUTPUT_NAME}"
            LIBRARY_OUTPUT_DIRECTORY "${LV2_BUNDLE_DIR}"
            # Look for libonnxruntime.so in the same directory as the plugin .so
            BUILD_RPATH   "\$ORIGIN"
            INSTALL_RPATH "\$ORIGIN"
        )
        target_compile_features(PiPitchImpl_${SUFFIX} PRIVATE cxx_std_17)
        target_compile_options(PiPitchImpl_${SUFFIX} PRIVATE
            -O3 -ffast-math -fPIC ${MARCH_FLAGS}
        )
        target_include_directories(PiPitchImpl_${SUFFIX} PRIVATE
            ${CMAKE_SOURCE_DIR}/LV2
            ${NN}/Lib/Model
            ${NN}/Lib/Utils
            ${NN}/ThirdParty/RTNeural
            ${NN}/ThirdParty/onnxruntime/include
            ${LV2_INCLUDE_DIRS}
        )
        target_compile_definitions(PiPitchImpl_${SUFFIX} PRIVATE
            SAVE_DOWNSAMPLED_AUDIO=0
            USE_TEST_NOTE_FRAME_TO_TIME=0
            "PIPITCH_IMPL_NAME=\"${OUTPUT_NAME}\""
            "LV2_WINDOW_MS=${LV2_WINDOW_MS}"
        )
        target_link_libraries(PiPitchImpl_${SUFFIX} PRIVATE
            RTNeural
            ${LV2_ONNXRUNTIME_LIB}
            ${LV2_LIBRARIES}
            pthread dl
        )
    endfunction()

    # ── Implementation .so files (one per SIMD tier) ──────────────────────────
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
        pipitch_add_impl(NEON   "pipitch_impl_neon"   -march=armv8-a+simd)
        pipitch_add_impl(ARMv82 "pipitch_impl_armv82" -march=armv8.2-a+dotprod+fp16)
        message(STATUS "LV2: building aarch64 impls: neon + armv8.2-a+dotprod+fp16")
    elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
        pipitch_add_impl(SSE42  "pipitch_impl_sse42"  -msse4.2)
        pipitch_add_impl(AVX2   "pipitch_impl_avx2"   -mavx2 -mfma)
        message(STATUS "LV2: building x86-64 impls: sse4.2 + avx2+fma")
    else()
        pipitch_add_impl(Baseline "pipitch_impl_baseline")
        message(STATUS "LV2: building generic baseline impl")
    endif()

    # ── Wrapper .so (the only target the LV2 host loads) ─────────────────────
    add_library(PiPitch_LV2 MODULE
        ${CMAKE_SOURCE_DIR}/LV2/pipitch.cpp
    )
    set_target_properties(PiPitch_LV2 PROPERTIES
        PREFIX ""
        OUTPUT_NAME "pipitch"
        LIBRARY_OUTPUT_DIRECTORY "${LV2_BUNDLE_DIR}"
    )
    target_compile_features(PiPitch_LV2 PRIVATE cxx_std_17)
    target_compile_options(PiPitch_LV2 PRIVATE -O2 -fPIC)
    target_include_directories(PiPitch_LV2 PRIVATE
        ${LV2_INCLUDE_DIRS}
    )
    target_link_libraries(PiPitch_LV2 PRIVATE
        ${LV2_LIBRARIES}
        dl
    )

    # ── Bundle assembly ───────────────────────────────────────────────────────
    add_custom_command(TARGET PiPitch_LV2 POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy
            ${CMAKE_SOURCE_DIR}/LV2/manifest.ttl
            ${LV2_BUNDLE_DIR}/manifest.ttl
        COMMAND ${CMAKE_COMMAND} -E copy
            ${CMAKE_SOURCE_DIR}/LV2/plugin.ttl
            ${LV2_BUNDLE_DIR}/plugin.ttl
        COMMAND ${CMAKE_COMMAND} -E copy
            ${CMAKE_SOURCE_DIR}/LV2/pipitch_ranges.conf
            ${LV2_BUNDLE_DIR}/pipitch_ranges.conf
        COMMAND ${CMAKE_COMMAND} -E copy_directory
            ${NN}/Lib/ModelData
            ${LV2_BUNDLE_DIR}/ModelData
        COMMAND ${CMAKE_COMMAND} -E copy
            ${CMAKE_SOURCE_DIR}/LV2/swift_f0_model.onnx
            ${LV2_BUNDLE_DIR}/swift_f0_model.onnx
        COMMAND ${CMAKE_COMMAND} -E copy
            ${LV2_ONNXRUNTIME_LIB}
            ${LV2_BUNDLE_DIR}/${LV2_ONNXRUNTIME_NAME}
        COMMAND ${CMAKE_COMMAND} -E create_symlink
            ${LV2_ONNXRUNTIME_NAME}
            ${LV2_BUNDLE_DIR}/libonnxruntime.so.1
        COMMENT "Assembling LV2 bundle: ${LV2_BUNDLE_DIR}"
    )

    message(STATUS "LV2 bundle will be assembled at: ${LV2_BUNDLE_DIR}")

    # ── Latency benchmark ─────────────────────────────────────────────────────
    option(BUILD_LV2_BENCH "Build the PiPitch latency benchmark" ON)
    if(BUILD_LV2_BENCH)
        add_executable(latency_bench
            ${CMAKE_SOURCE_DIR}/LV2/latency_bench.cpp
            ${LIB_MODEL_SOURCES}
        )
        set_target_properties(latency_bench PROPERTIES
            BUILD_RPATH   "${LV2_BUNDLE_DIR}"
            INSTALL_RPATH "\$ORIGIN/${LV2_BUNDLE_NAME}"
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
        add_executable(pipitch_tune
            ${CMAKE_SOURCE_DIR}/LV2/pipitch_tune.cpp
            ${LIB_MODEL_SOURCES}
        )
        set_target_properties(pipitch_tune PROPERTIES
            BUILD_RPATH   "${LV2_BUNDLE_DIR}"
            INSTALL_RPATH "\$ORIGIN/${LV2_BUNDLE_NAME}"
        )
        target_compile_features(pipitch_tune PRIVATE cxx_std_17)
        target_compile_options(pipitch_tune PRIVATE -O2 -fPIC)
        target_include_directories(pipitch_tune PRIVATE
            ${CMAKE_SOURCE_DIR}/LV2
            ${NN}/Lib/Model
            ${NN}/Lib/Utils
            ${NN}/ThirdParty/RTNeural
            ${NN}/ThirdParty/onnxruntime/include
        )
        target_compile_definitions(pipitch_tune PRIVATE
            SAVE_DOWNSAMPLED_AUDIO=0
            USE_TEST_NOTE_FRAME_TO_TIME=0
        )
        find_library(FFTW3F_LIB fftw3f)
        target_link_libraries(pipitch_tune PRIVATE
            RTNeural
            ${LV2_ONNXRUNTIME_LIB}
            ${JACK_LIB}
            $<$<BOOL:${FFTW3F_LIB}>:${FFTW3F_LIB}>
            pthread dl
        )
        message(STATUS "LV2: pipitch_tune target enabled")
    endif()

    option(BUILD_LV2_TEST "Build the record/test regression tool (requires JACK)" ON)
    if(BUILD_LV2_TEST)
        if(NOT JACK_LIB)
            find_library(JACK_LIB jack REQUIRED)
        endif()
        add_executable(pipitch_test
            ${CMAKE_SOURCE_DIR}/LV2/pipitch_test.cpp
            ${LIB_MODEL_SOURCES}
        )
        set_target_properties(pipitch_test PROPERTIES
            BUILD_RPATH   "${LV2_BUNDLE_DIR}"
            INSTALL_RPATH "\$ORIGIN/${LV2_BUNDLE_NAME}"
        )
        target_compile_features(pipitch_test PRIVATE cxx_std_17)
        target_compile_options(pipitch_test PRIVATE -O2 -fPIC)
        target_include_directories(pipitch_test PRIVATE
            ${CMAKE_SOURCE_DIR}/LV2
            ${NN}/Lib/Model
            ${NN}/Lib/Utils
            ${NN}/ThirdParty/RTNeural
            ${NN}/ThirdParty/onnxruntime/include
        )
        target_compile_definitions(pipitch_test PRIVATE
            SAVE_DOWNSAMPLED_AUDIO=0
            USE_TEST_NOTE_FRAME_TO_TIME=0
        )
        if(NOT FFTW3F_LIB)
            find_library(FFTW3F_LIB fftw3f)
        endif()
        target_link_libraries(pipitch_test PRIVATE
            RTNeural
            ${LV2_ONNXRUNTIME_LIB}
            ${JACK_LIB}
            $<$<BOOL:${FFTW3F_LIB}>:${FFTW3F_LIB}>
            pthread dl
        )
        message(STATUS "LV2: pipitch_test target enabled")
    endif()

    # ── Install targets ───────────────────────────────────────────────────────
    set(LV2_INSTALL_PATH "/zynthian/zynthian-plugins/lv2"
        CACHE STRING "Destination for the LV2 bundle (default: Zynthian LV2 path)")

    install(DIRECTORY "${LV2_BUNDLE_DIR}/"
        DESTINATION "${LV2_INSTALL_PATH}/${LV2_BUNDLE_NAME}"
        USE_SOURCE_PERMISSIONS
    )

    install(PROGRAMS "${CMAKE_SOURCE_DIR}/LV2/pipitch-connect.sh"
        DESTINATION /usr/local/bin
        RENAME pipitch-connect.sh
    )

    if(BUILD_LV2_TUNE)
        install(TARGETS pipitch_tune
            RUNTIME DESTINATION /usr/local/bin
        )
    endif()

    install(FILES "${CMAKE_SOURCE_DIR}/LV2/pipitch-connect.service"
        DESTINATION /etc/systemd/system
    )

    install(CODE "
        message(STATUS \"-- Running systemctl daemon-reload\")
        execute_process(COMMAND systemctl daemon-reload)

        execute_process(
            COMMAND systemctl is-enabled pipitch-connect.service
            OUTPUT_VARIABLE _svc_enabled OUTPUT_STRIP_TRAILING_WHITESPACE
            RESULT_VARIABLE _svc_rc
        )
        if(NOT _svc_enabled STREQUAL \"enabled\")
            message(STATUS \"-- Enabling pipitch-connect.service\")
            execute_process(COMMAND systemctl enable pipitch-connect.service)
        else()
            message(STATUS \"-- pipitch-connect.service already enabled\")
        endif()

        message(STATUS \"-- Restarting zynthian.service\")
        execute_process(COMMAND systemctl restart zynthian.service)

        message(STATUS \"-- Starting pipitch-connect.service\")
        execute_process(COMMAND systemctl restart pipitch-connect.service)
    ")

endif()
