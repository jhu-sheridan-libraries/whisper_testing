#ifndef WHISPER_DIARIZE_H
#define WHISPER_DIARIZE_H

#include <string>
#include <vector>
#include <memory>
#include "whisper.h"
#include "benchmark.h"

enum class OutputFormat {
    VTT,
    TXT
};

struct TranscriptionOptions {
    std::string model_path;
    std::string language = "auto";
    int num_speakers = 0; // 0 means auto-detect
    OutputFormat format = OutputFormat::VTT;
};

struct Segment {
    int64_t start_ms;
    int64_t end_ms;
    int speaker_id;
    std::string text;
};

class WhisperDiarize {
public:
    WhisperDiarize();
    ~WhisperDiarize();

    bool initialize(const TranscriptionOptions& options);
    bool transcribe(const std::string& audio_path, const std::string& output_path);
    
    // Getters for benchmark results
    double getProcessingTime() const;
    double getRealTimeFactor() const;

private:
    struct whisper_context* ctx = nullptr;
    TranscriptionOptions options;
    std::vector<Segment> segments;
    std::unique_ptr<Benchmark> benchmark;
    
    bool loadAudio(const std::string& audio_path, std::vector<float>& pcmf32, int& audio_length_s);
    bool runWhisperInference(const std::vector<float>& pcmf32);
    bool detectSpeakers(int num_speakers);
    bool writeOutput(const std::string& output_path);
    
    void cleanupWhisperContext();
};

#endif // WHISPER_DIARIZE_H 