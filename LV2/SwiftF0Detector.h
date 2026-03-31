#pragma once
/**
 * SwiftF0Detector — monophonic pitch estimation via the SwiftF0 ONNX model.
 *
 * Model: swift_f0_model.onnx (389 KB)
 * Input:  "input_audio"  shape (1, N)  float32  at 16 kHz
 * Outputs: "pitch_hz"   shape (1, F)  float32  F = N / 256 frames
 *          "confidence" shape (1, F)  float32
 *
 * infer() returns the MIDI note (rounded integer) of the median confident
 * frame, or -1 if no frame exceeds the confidence threshold.
 *
 * The model accepts any audio length; at 16 kHz, 80 ms = 1280 samples → 5 frames,
 * 150 ms = 2400 samples → 9 frames.  Confidence 0.5 is a reasonable default for
 * guitar; raise to 0.7–0.8 to suppress false detections in noisy conditions.
 */

#include <algorithm>
#include <cmath>
#include <memory>
#include <string>
#include <vector>

#include "onnxruntime_cxx_api.h"

class SwiftF0Detector {
public:
    // Loads model from modelPath.  Throws std::exception if file not found
    // or session creation fails.
    explicit SwiftF0Detector(const std::string& modelPath)
        : env_(ORT_LOGGING_LEVEL_WARNING, "SwiftF0")
    {
        opts_.SetIntraOpNumThreads(1);
        opts_.SetInterOpNumThreads(1);
        opts_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        session_ = std::make_unique<Ort::Session>(env_, modelPath.c_str(), opts_);
    }

    // Run inference.
    // audio16k  : float32 samples at 16 kHz
    // nSamples  : sample count (must be >= 256 for at least one frame)
    // threshold : minimum confidence to accept a frame (0–1, default 0.5)
    // outHz     : if non-null, receives the median Hz of confident frames
    // outMaxConf: if non-null, receives the maximum confidence across all frames
    // Returns   : MIDI note (integer, rounded) of the median confident frame,
    //             or -1 if no frame passes the threshold.
    int infer(const float* audio16k, int nSamples, float threshold = 0.5f,
              float* outHz = nullptr, float* outMaxConf = nullptr)
    {
        if (outHz)      *outHz = -1.0f;
        if (outMaxConf) *outMaxConf = 0.0f;
        if (nSamples < 256) return -1;

        const int64_t n = static_cast<int64_t>(nSamples);
        std::array<int64_t, 2> inShape = {1, n};

        Ort::MemoryInfo memInfo = Ort::MemoryInfo::CreateCpu(
            OrtArenaAllocator, OrtMemTypeDefault);
        Ort::Value inTensor = Ort::Value::CreateTensor<float>(
            memInfo,
            const_cast<float*>(audio16k), static_cast<size_t>(nSamples),
            inShape.data(), inShape.size());

        const char* inputNames[]  = {"input_audio"};
        const char* outputNames[] = {"pitch_hz", "confidence"};

        auto outputs = session_->Run(
            Ort::RunOptions{nullptr},
            inputNames,  &inTensor, 1,
            outputNames, 2);

        const float* pitchHz = outputs[0].GetTensorData<float>();
        const float* conf    = outputs[1].GetTensorData<float>();

        const auto shape = outputs[0].GetTensorTypeAndShapeInfo().GetShape();
        const int nFrames = static_cast<int>(
            shape.size() >= 2 ? shape[1] : (shape.empty() ? 0 : shape[0]));
        if (nFrames <= 0) return -1;

        // Track max confidence across all frames (for pitch bend stability gate).
        float maxConf = 0.0f;
        for (int i = 0; i < nFrames; ++i)
            if (conf[i] > maxConf) maxConf = conf[i];
        if (outMaxConf) *outMaxConf = maxConf;

        // Collect pitches from frames whose confidence exceeds the threshold.
        hz_.clear();
        for (int i = 0; i < nFrames; ++i)
            if (conf[i] >= threshold && pitchHz[i] > 20.0f)
                hz_.push_back(pitchHz[i]);

        if (hz_.empty()) return -1;

        // Median pitch avoids outlier frames pulling the result sharp or flat.
        std::sort(hz_.begin(), hz_.end());
        const float medHz = hz_[hz_.size() / 2];
        if (outHz) *outHz = medHz;

        // Hz → MIDI (equal temperament, A4 = 440 Hz = MIDI 69)
        const float midi = 69.0f + 12.0f * std::log2(medHz / 440.0f);
        return static_cast<int>(std::round(midi));
    }

private:
    Ort::Env            env_;
    Ort::SessionOptions opts_;
    std::unique_ptr<Ort::Session> session_;
    std::vector<float>  hz_;   // scratch buffer; avoids per-call allocation
};
