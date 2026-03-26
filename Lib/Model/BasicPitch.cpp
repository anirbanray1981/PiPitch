//
// Created by Damien Ronssin on 10.03.23.
//

#include "BasicPitch.h"

void BasicPitch::reset()
{
    mBasicPitchCNN.reset();
    mNotesCreator.clear();

    mContoursPG.clear();
    mContoursPG.shrink_to_fit();
    mNotesPG.clear();
    mNotesPG.shrink_to_fit();
    mOnsetsPG.clear();
    mOnsetsPG.shrink_to_fit();
    mNoteEvents.clear();
    mNoteEvents.shrink_to_fit();

    mNumFrames = 0;
    mHasState  = false;
}

void BasicPitch::setParameters(float inNoteSensitivity, float inSplitSensitivity, float inMinNoteDurationMs)
{
    mParams.frameThreshold = 1.0f - inNoteSensitivity;
    mParams.onsetThreshold = 1.0f - inSplitSensitivity;

    mParams.minNoteLength =
        static_cast<int>(std::round(inMinNoteDurationMs / 1000.0f / (FFT_HOP / BASIC_PITCH_SAMPLE_RATE)));

    mParams.pitchBend = MultiPitchBend;
    mParams.melodiaTrick = true;
    mParams.inferOnsets = true;
}

void BasicPitch::transcribeToMIDI(float* inAudio, int inNumSamples)
{
    // To test if downsampling works as expected
#if SAVE_DOWNSAMPLED_AUDIO
    auto file = juce::File::getSpecialLocation(juce::File::userDesktopDirectory).getChildFile("Test_Downsampled.wav");

    std::unique_ptr<AudioFormatWriter> format_writer;

    format_writer.reset(WavAudioFormat().createWriterFor(new FileOutputStream(file), 22050, 1, 16, {}, 0));

    if (format_writer != nullptr) {
        AudioBuffer<float> tmp_buffer;
        tmp_buffer.setSize(1, inNumSamples);
        tmp_buffer.copyFrom(0, 0, inAudio, inNumSamples);
        format_writer->writeFromAudioSampleBuffer(tmp_buffer, 0, inNumSamples);

        format_writer->flush();

        file.revealToUser();
    }
#endif

    const float* stacked_cqt = mFeaturesCalculator.computeFeatures(inAudio, inNumSamples, mNumFrames);

    if (mNumFrames == 0) {
        mNoteEvents.clear();
        return;
    }

    mOnsetsPG.resize(mNumFrames, std::vector<float>(static_cast<size_t>(NUM_FREQ_OUT), 0.0f));
    mNotesPG.resize(mNumFrames, std::vector<float>(static_cast<size_t>(NUM_FREQ_OUT), 0.0f));
    mContoursPG.resize(mNumFrames, std::vector<float>(static_cast<size_t>(NUM_FREQ_IN), 0.0f));

    mOnsetsPG.shrink_to_fit();
    mNotesPG.shrink_to_fit();
    mContoursPG.shrink_to_fit();

    // Reset Conv2D internal state. If a prior call's circular buffer state was
    // saved, restore it immediately so the warmup sees continuous audio context
    // rather than a zero baseline. This replaces the old zero-warmup phase (which
    // was a no-op after reset()) and removes the hard ~116 ms minimum constraint.
    mBasicPitchCNN.reset();
    if (mHasState)
        mBasicPitchCNN.restoreCircularState(mSavedState);

    const size_t num_lh_frames = static_cast<size_t>(BasicPitchCNN::getNumFramesLookahead());
    const std::vector<float> zero_stacked_cqt(NUM_HARMONICS * NUM_FREQ_IN, 0.0f);

    // Dummy output slots for frames whose output index falls outside [0, mNumFrames).
    std::vector<float> dummy_contour(static_cast<size_t>(NUM_FREQ_IN),  0.0f);
    std::vector<float> dummy_notes  (static_cast<size_t>(NUM_FREQ_OUT), 0.0f);
    std::vector<float> dummy_onsets (static_cast<size_t>(NUM_FREQ_OUT), 0.0f);

    // Phases 2+3 unified: feed all real frames through the CNN.
    // Frames 0..(num_lh_frames-1) warm up the Conv2D history buffers (outputs
    // fall outside the valid window and are directed to dummy vectors).
    // Frames num_lh_frames..(mNumFrames-1) write to the real output arrays.
    // Short windows (mNumFrames < num_lh_frames) are handled gracefully: all
    // frames go to dummy outputs, and the zero-tail phase below covers the rest.
    for (size_t frame_idx = 0; frame_idx < mNumFrames; frame_idx++) {
        const int out_idx = static_cast<int>(frame_idx) - static_cast<int>(num_lh_frames);
        const bool valid  = out_idx >= 0 && static_cast<size_t>(out_idx) < mNumFrames;
        mBasicPitchCNN.frameInference(stacked_cqt + frame_idx * NUM_HARMONICS * NUM_FREQ_IN,
                                      valid ? mContoursPG[out_idx] : dummy_contour,
                                      valid ? mNotesPG[out_idx]    : dummy_notes,
                                      valid ? mOnsetsPG[out_idx]   : dummy_onsets);
    }

    // Save circular state here — before the zero-tail — so the next call can
    // restore it and skip warmup phases, cutting CNN calls by ~31%.
    mBasicPitchCNN.saveCircularState(mSavedState);
    mHasState = true;

    // Phase 4 (zero-tail): flush the last num_lh_frames outputs from the pipeline.
    for (size_t frame_idx = mNumFrames; frame_idx < mNumFrames + num_lh_frames; frame_idx++) {
        const int out_idx = static_cast<int>(frame_idx) - static_cast<int>(num_lh_frames);
        const bool valid  = out_idx >= 0 && static_cast<size_t>(out_idx) < mNumFrames;
        mBasicPitchCNN.frameInference(zero_stacked_cqt.data(),
                                      valid ? mContoursPG[out_idx] : dummy_contour,
                                      valid ? mNotesPG[out_idx]    : dummy_notes,
                                      valid ? mOnsetsPG[out_idx]   : dummy_onsets);
    }

    mNoteEvents = mNotesCreator.convert(mNotesPG, mOnsetsPG, mContoursPG, mParams, true);
}

void BasicPitch::updateMIDI()
{
    mNoteEvents = mNotesCreator.convert(mNotesPG, mOnsetsPG, mContoursPG, mParams, false);
}

const std::vector<Notes::Event>& BasicPitch::getNoteEvents() const
{
    return mNoteEvents;
}
