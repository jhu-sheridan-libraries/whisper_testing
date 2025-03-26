#pragma once

#include <string>
#include <vector>
#include <chrono>
#include <optional>
#include <cstdint>
#include "whisper.h"

struct TranscriptionSegment {
    double start;
    double end;
    std::string text;
    std::string speaker;
};

struct BenchmarkData {
    std::string timestamp;
    std::string filename;
    double duration_seconds;
    int num_speakers;
    std::string model;
    double processing_time_seconds;
    double real_time_factor;
    double cpu_percent;
    int cpu_count;
    double memory_percent;
};

struct ProcessingEstimate {
    double total_minutes;
    double transcription_minutes;
    double diarization_minutes;
    double real_time_factor;
};

class WhisperDiarize {
public:
    WhisperDiarize();
    ~WhisperDiarize();

    // Initialize with model path and parameters
    bool init(const std::string& model_path, const std::string& model_size);
    
    // Main processing functions
    std::vector<TranscriptionSegment> process_audio(
        const std::string& audio_path,
        const std::string& output_path,
        const std::string& format = "vtt",
        int num_speakers = 0,
        const std::string& language = "",
        const std::string& task = "transcribe"
    );

    // Utility functions
    static double get_audio_duration(const std::string& audio_path);
    static ProcessingEstimate estimate_processing_time(
        double duration,
        const std::string& model,
        std::optional<int> num_speakers
    );
    
    // Save output in different formats
    void save_vtt(const std::vector<TranscriptionSegment>& segments, const std::string& output_path);
    void save_txt(const std::vector<TranscriptionSegment>& segments, const std::string& output_path);
    
    // Benchmark logging
    void log_benchmark(const BenchmarkData& data);

private:
    // Internal whisper context
    struct whisper_context* ctx;
    std::string model_size;
    
    // Helper functions
    std::string format_timestamp(double seconds);
    std::vector<TranscriptionSegment> transcribe_audio(const std::string& audio_path);
    std::vector<TranscriptionSegment> perform_diarization(
        const std::string& audio_path,
        std::optional<int> num_speakers
    );
}; 