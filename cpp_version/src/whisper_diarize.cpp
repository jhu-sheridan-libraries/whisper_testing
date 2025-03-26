#include "whisper_diarize.hpp"
#include "whisper.h"
#include <fstream>
#include <iostream>
#include <ctime>
#include <sstream>
#include <filesystem>
#include <thread>
#include <map>
#include <optional>
#include <iomanip>
#include <cmath>
#define DR_WAV_IMPLEMENTATION
#include "third_party/dr_wav.h"
#include <sys/resource.h>
#include <sys/sysinfo.h>
#include "benchmark.h"

WhisperDiarize::WhisperDiarize() : ctx(nullptr), benchmark(std::make_unique<Benchmark>()) {}

WhisperDiarize::~WhisperDiarize() {
    if (ctx) {
        whisper_free(ctx);
    }
}

bool WhisperDiarize::initialize(const TranscriptionOptions& options) {
    this->options = options;
    
    // Initialize whisper context with parameters
    struct whisper_context_params cparams = whisper_context_default_params();
    ctx = whisper_init_from_file_with_params(options.model_path.c_str(), cparams);
    
    if (!ctx) {
        std::cerr << "Failed to initialize whisper context\n";
        return false;
    }
    
    return true;
}

bool WhisperDiarize::transcribe(const std::string& audio_path, const std::string& output_path) {
    std::vector<float> pcmf32;
    int audio_length_s = 0;
    
    // Start benchmarking
    benchmark->start();
    
    if (!loadAudio(audio_path, pcmf32, audio_length_s)) {
        return false;
    }
    
    if (!runWhisperInference(pcmf32)) {
        return false;
    }
    
    if (!detectSpeakers(options.num_speakers)) {
        return false;
    }
    
    if (!writeOutput(output_path)) {
        return false;
    }
    
    // Stop benchmarking and record results
    std::string model_name = options.model_path.substr(options.model_path.find_last_of("/\\") + 1);
    benchmark->stop(audio_path, audio_length_s, options.num_speakers, model_name);
    
    return true;
}

double WhisperDiarize::getProcessingTime() const {
    return benchmark->getProcessingTime();
}

double WhisperDiarize::getRealTimeFactor() const {
    return benchmark->getRealTimeFactor();
}

bool WhisperDiarize::loadAudio(const std::string& audio_path, std::vector<float>& pcmf32, int& audio_length_s) {
    // Load audio file
    drwav wav;
    if (!drwav_init_file(&wav, audio_path.c_str(), nullptr)) {
        std::cerr << "Failed to open WAV file: " << audio_path << std::endl;
        return false;
    }
    
    // Allocate memory for PCM data
    pcmf32.resize(wav.totalPCMFrameCount);
    
    // Read PCM data
    drwav_read_pcm_frames_f32(&wav, wav.totalPCMFrameCount, pcmf32.data());
    
    // Calculate audio length in seconds
    audio_length_s = wav.totalPCMFrameCount / wav.sampleRate;
    
    // Clean up
    drwav_uninit(&wav);
    
    return true;
}

bool WhisperDiarize::runWhisperInference(const std::vector<float>& pcmf32) {
    // Set up whisper parameters
    whisper_full_params wparams = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);
    
    // Set language if specified
    if (options.language != "auto") {
        wparams.language = options.language.c_str();
    }
    
    // Run inference
    if (whisper_full(ctx, wparams, pcmf32.data(), pcmf32.size()) != 0) {
        std::cerr << "Failed to run whisper inference" << std::endl;
        return false;
    }
    
    // Extract segments
    int n_segments = whisper_full_n_segments(ctx);
    segments.clear();
    
    for (int i = 0; i < n_segments; ++i) {
        Segment segment;
        segment.start_ms = whisper_full_get_segment_t0(ctx, i) * 10;
        segment.end_ms = whisper_full_get_segment_t1(ctx, i) * 10;
        segment.speaker_id = 0; // Default speaker ID, will be updated in detectSpeakers
        segment.text = whisper_full_get_segment_text(ctx, i);
        
        segments.push_back(segment);
    }
    
    return true;
}

bool WhisperDiarize::detectSpeakers(int num_speakers) {
    // Simple speaker diarization based on time gaps
    // In a real implementation, this would use a more sophisticated algorithm
    
    if (segments.empty()) {
        return true;
    }
    
    // If num_speakers is 0 or 1, assign all to speaker 0
    if (num_speakers <= 1) {
        for (auto& segment : segments) {
            segment.speaker_id = 0;
        }
        return true;
    }
    
    // Simple heuristic: assign speakers based on time gaps
    int current_speaker = 0;
    segments[0].speaker_id = current_speaker;
    
    for (size_t i = 1; i < segments.size(); ++i) {
        // If there's a significant gap, switch speakers
        if (segments[i].start_ms - segments[i-1].end_ms > 1000) {
            current_speaker = (current_speaker + 1) % num_speakers;
        }
        segments[i].speaker_id = current_speaker;
    }
    
    return true;
}

bool WhisperDiarize::writeOutput(const std::string& output_path) {
    if (segments.empty()) {
        std::cerr << "No segments to write" << std::endl;
        return false;
    }
    
    std::ofstream out(output_path);
    if (!out.is_open()) {
        std::cerr << "Failed to open output file: " << output_path << std::endl;
        return false;
    }
    
    if (options.format == OutputFormat::VTT) {
        // Write WebVTT format
        out << "WEBVTT\n\n";
        
        for (const auto& segment : segments) {
            // Format timestamps as HH:MM:SS.mmm
            auto format_timestamp = [](int64_t ms) {
                int hours = ms / 3600000;
                int minutes = (ms % 3600000) / 60000;
                int seconds = (ms % 60000) / 1000;
                int milliseconds = ms % 1000;
                
                std::stringstream ss;
                ss << std::setfill('0') << std::setw(2) << hours << ":"
                   << std::setfill('0') << std::setw(2) << minutes << ":"
                   << std::setfill('0') << std::setw(2) << seconds << "."
                   << std::setfill('0') << std::setw(3) << milliseconds;
                return ss.str();
            };
            
            out << format_timestamp(segment.start_ms) << " --> " << format_timestamp(segment.end_ms) << "\n";
            out << "<v Speaker " << segment.speaker_id << ">" << segment.text << "\n\n";
        }
    } else {
        // Write plain text format
        for (const auto& segment : segments) {
            out << "[Speaker " << segment.speaker_id << "] " << segment.text << "\n";
        }
    }
    
    out.close();
    return true;
}

void WhisperDiarize::cleanupWhisperContext() {
    if (ctx) {
        whisper_free(ctx);
        ctx = nullptr;
    }
}

std::vector<TranscriptionSegment> WhisperDiarize::process_audio(
    const std::string& audio_path,
    const std::string& output_path,
    const std::string& format,
    int num_speakers,
    const std::string& language,
    const std::string& task
) {
    auto start_time = std::chrono::steady_clock::now();
    
    // Get audio duration
    double duration = get_audio_duration(audio_path);
    if (duration <= 0) {
        throw std::runtime_error("Failed to get audio duration for: " + audio_path);
    }
    std::cout << "Audio duration: " << duration << " seconds\n";
    
    // Transcribe audio
    auto segments = transcribe_audio(audio_path);
    if (segments.empty()) {
        throw std::runtime_error("No transcription segments produced");
    }
    
    // Save output
    if (format == "vtt") {
        save_vtt(segments, output_path);
    } else if (format == "txt") {
        save_txt(segments, output_path);
    }
    
    // Calculate processing time
    auto end_time = std::chrono::steady_clock::now();
    double processing_time = std::chrono::duration<double>(end_time - start_time).count();
    
    // Create benchmark data
    BenchmarkData benchmark{
        get_current_timestamp(),                      // timestamp
        std::filesystem::path(audio_path).filename().string(), // filename
        duration,                                     // duration_seconds
        num_speakers,                                 // num_speakers
        this->model_size,                            // model
        processing_time,                              // processing_time_seconds
        processing_time / duration,                   // real_time_factor
        0.0,                                         // cpu_percent (will be set in log_benchmark)
        std::thread::hardware_concurrency(),         // cpu_count
        0.0                                          // memory_percent (will be set in log_benchmark)
    };
    
    log_benchmark(benchmark);
    
    return segments;
}

std::string WhisperDiarize::get_current_timestamp() {
    auto now = std::chrono::system_clock::now();
    auto now_time_t = std::chrono::system_clock::to_time_t(now);
    std::stringstream ss;
    ss << std::put_time(std::localtime(&now_time_t), "%Y-%m-%d %H:%M:%S");
    return ss.str();
}

std::vector<float> load_wav_file(const std::string& fname) {
    drwav wav;
    std::vector<float> pcmf32;

    if (!drwav_init_file(&wav, fname.c_str(), NULL)) {
        fprintf(stderr, "Failed to open WAV file '%s'\n", fname.c_str());
        return {};
    }

    if (wav.channels != 1) {
        fprintf(stderr, "WAV file '%s' must be mono\n", fname.c_str());
        drwav_uninit(&wav);
        return {};
    }

    if (wav.sampleRate != 16000) {
        fprintf(stderr, "WAV file '%s' must be 16kHz\n", fname.c_str());
        drwav_uninit(&wav);
        return {};
    }

    size_t n = wav.totalPCMFrameCount;
    pcmf32.resize(n);
    drwav_read_pcm_frames_f32(&wav, n, pcmf32.data());
    drwav_uninit(&wav);

    // Print some debug info
    printf("Loaded WAV file: %s\n", fname.c_str());
    printf("  Sample rate: %d Hz\n", wav.sampleRate);
    printf("  Channels: %d\n", wav.channels);
    printf("  Duration: %.2f seconds\n", float(n)/wav.sampleRate);

    return pcmf32;
}

std::vector<TranscriptionSegment> WhisperDiarize::transcribe_audio(const std::string& audio_path) {
    std::vector<TranscriptionSegment> segments;
    
    // Load audio file
    std::vector<float> pcmf32 = load_wav_file(audio_path);
    if (pcmf32.empty()) {
        throw std::runtime_error("Failed to load audio file: " + audio_path);
    }
    
    std::cout << "DEBUG: Processing " << pcmf32.size() << " samples\n";
    
    // Process audio with whisper
    whisper_full_params wparams = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);
    
    // Set parameters
    wparams.print_progress   = true;
    wparams.print_special    = false;
    wparams.print_realtime   = false;
    wparams.print_timestamps = true;
    wparams.translate        = false;
    wparams.language        = "en";
    wparams.n_threads       = std::thread::hardware_concurrency();
    wparams.offset_ms       = 0;
    wparams.duration_ms     = 0;
    
    std::cout << "DEBUG: Starting whisper_full\n";
    if (whisper_full(ctx, wparams, pcmf32.data(), pcmf32.size()) != 0) {
        throw std::runtime_error("Failed to process audio");
    }
    
    // Extract segments
    const int n_segments = whisper_full_n_segments(ctx);
    std::cout << "DEBUG: Found " << n_segments << " segments\n";
    
    for (int i = 0; i < n_segments; ++i) {
        const char* text = whisper_full_get_segment_text(ctx, i);
        double t0 = whisper_full_get_segment_t0(ctx, i) / 100.0;
        double t1 = whisper_full_get_segment_t1(ctx, i) / 100.0;
        
        std::cout << "DEBUG: Segment " << i << ": " << t0 << "s -> " << t1 << "s: " << text << "\n";
        
        TranscriptionSegment segment{
            t0,
            t1,
            text,
            "SPEAKER_01"  // Default speaker
        };
        segments.push_back(segment);
    }
    
    std::cout << "DEBUG: Transcription complete, returning " << segments.size() << " segments\n";
    return segments;
}

double WhisperDiarize::get_audio_duration(const std::string& audio_path) {
    // Use ffprobe to get duration
    std::string cmd = "ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 \"" + audio_path + "\"";
    
    FILE* pipe = popen(cmd.c_str(), "r");
    if (!pipe) {
        std::cerr << "Error executing ffprobe\n";
        return 0.0;
    }
    
    char buffer[128];
    std::string result = "";
    while (!feof(pipe)) {
        if (fgets(buffer, 128, pipe) != NULL)
            result += buffer;
    }
    pclose(pipe);
    
    try {
        return std::stod(result);
    } catch (...) {
        std::cerr << "Error parsing duration: " << result << "\n";
        return 0.0;
    }
}

ProcessingEstimate WhisperDiarize::estimate_processing_time(
    double duration,
    const std::string& model,
    std::optional<int> num_speakers
) {
    // Base real-time factors for transcription
    std::map<std::string, double> model_factors = {
        {"tiny", 0.8},
        {"base", 1.5},
        {"small", 2.5},
        {"medium", 3.5},
        {"large", 5.0},
        {"large-v2", 6.0}
    };

    // Diarization adds overhead
    double diarization_factor = num_speakers.has_value() ? 1.5 : 0;
    
    // Calculate estimates
    double transcription_time = duration * model_factors[model];
    double diarization_time = duration * diarization_factor;
    double total_time = transcription_time + diarization_time;
    
    return ProcessingEstimate{
        total_time / 60.0,           // total_minutes
        transcription_time / 60.0,   // transcription_minutes
        diarization_time / 60.0,     // diarization_minutes
        total_time / duration        // real_time_factor
    };
}

void WhisperDiarize::save_vtt(
    const std::vector<TranscriptionSegment>& segments,
    const std::string& output_path
) {
    std::ofstream out(output_path);
    if (!out.is_open()) {
        throw std::runtime_error("Failed to open output file: " + output_path);
    }

    out << "WEBVTT\n\n";

    for (size_t i = 0; i < segments.size(); i++) {
        const auto& segment = segments[i];
        
        // Format timestamps as HH:MM:SS.mmm
        auto format_time = [](double seconds) {
            int hours = static_cast<int>(seconds) / 3600;
            int minutes = (static_cast<int>(seconds) % 3600) / 60;
            double secs = fmod(seconds, 60.0);
            std::stringstream ss;
            ss << std::setfill('0') << std::setw(2) << hours << ":"
               << std::setfill('0') << std::setw(2) << minutes << ":"
               << std::fixed << std::setprecision(3) << secs;
            return ss.str();
        };

        out << i + 1 << "\n";
        out << format_time(segment.start) << " --> " << format_time(segment.end) << "\n";
        out << "<v " << segment.speaker << ">" << segment.text << "\n\n";
    }
}

void WhisperDiarize::save_txt(
    const std::vector<TranscriptionSegment>& segments,
    const std::string& output_path
) {
    std::ofstream out(output_path);
    if (!out.is_open()) {
        throw std::runtime_error("Failed to open output file: " + output_path);
    }

    for (const auto& segment : segments) {
        out << "[" << segment.speaker << "] " << segment.text << "\n";
    }
}

std::vector<TranscriptionSegment> WhisperDiarize::perform_diarization(
    const std::string& audio_path,
    std::optional<int> num_speakers
) {
    // TODO: Implement speaker diarization
    // For now, return empty vector
    return std::vector<TranscriptionSegment>();
}

void WhisperDiarize::log_benchmark(const BenchmarkData& data) {
    // Validate input data
    if (data.timestamp.empty()) {
        throw std::runtime_error("Benchmark data missing timestamp");
    }
    if (data.filename.empty()) {
        throw std::runtime_error("Benchmark data missing filename");
    }
    if (data.duration_seconds <= 0) {
        throw std::runtime_error("Invalid duration: " + std::to_string(data.duration_seconds));
    }
    if (data.model.empty()) {
        throw std::runtime_error("Benchmark data missing model name");
    }
    if (data.processing_time_seconds <= 0) {
        throw std::runtime_error("Invalid processing time: " + std::to_string(data.processing_time_seconds));
    }
    if (data.cpu_count <= 0) {
        throw std::runtime_error("Invalid CPU count: " + std::to_string(data.cpu_count));
    }

    // Get CPU usage
    struct rusage usage;
    if (getrusage(RUSAGE_SELF, &usage) != 0) {
        throw std::runtime_error("Failed to get CPU usage metrics");
    }
    double cpu_percent = (usage.ru_utime.tv_sec + usage.ru_stime.tv_sec) * 100.0 / data.processing_time_seconds;
    
    // Get memory usage
    struct sysinfo si;
    if (sysinfo(&si) != 0) {
        throw std::runtime_error("Failed to get memory usage metrics");
    }
    
    double total_ram = si.totalram * si.mem_unit;
    double used_ram = (si.totalram - si.freeram) * si.mem_unit;
    if (total_ram <= 0) {
        throw std::runtime_error("Invalid total RAM: " + std::to_string(total_ram));
    }
    double memory_percent = (used_ram / total_ram) * 100.0;
    
    // Update the benchmark data with actual metrics
    BenchmarkData updated_data = data;
    updated_data.cpu_percent = cpu_percent;
    updated_data.memory_percent = memory_percent;
    
    // Validate the metrics
    if (updated_data.cpu_percent < 0 || updated_data.cpu_percent > 100) {
        throw std::runtime_error("Invalid CPU percentage: " + std::to_string(updated_data.cpu_percent));
    }
    if (updated_data.memory_percent < 0 || updated_data.memory_percent > 100) {
        throw std::runtime_error("Invalid memory percentage: " + std::to_string(updated_data.memory_percent));
    }
    
    std::cout << "DEBUG: Writing benchmark data to CSV with:\n"
              << "  Model: " << updated_data.model << "\n"
              << "  Duration: " << updated_data.duration_seconds << "s\n"
              << "  Processing time: " << updated_data.processing_time_seconds << "s\n"
              << "  Real-time factor: " << updated_data.real_time_factor << "\n"
              << "  CPU Usage: " << updated_data.cpu_percent << "%\n"
              << "  Memory Usage: " << updated_data.memory_percent << "%\n";
              
    std::string benchmark_file = std::filesystem::current_path().string() + "/whisper_benchmarks.csv";
    std::cout << "DEBUG: Writing to benchmark file: " << benchmark_file << "\n";
    
    std::ofstream out(benchmark_file, std::ios::app);
    if (!out.is_open()) {
        throw std::runtime_error("Failed to open benchmark file: " + benchmark_file);
    }

    // Write header if file is empty
    if (out.tellp() == 0) {
        out << "timestamp,filename,duration_seconds,num_speakers,model,"
            << "processing_time_seconds,real_time_factor,cpu_percent,"
            << "cpu_count,memory_percent\n";
    }

    // Write the data
    out << updated_data.timestamp << ","
        << updated_data.filename << ","
        << updated_data.duration_seconds << ","
        << updated_data.num_speakers << ","
        << updated_data.model << ","
        << updated_data.processing_time_seconds << ","
        << updated_data.real_time_factor << ","
        << updated_data.cpu_percent << ","
        << updated_data.cpu_count << ","
        << updated_data.memory_percent << "\n";
        
    out.close();
    std::cout << "DEBUG: Successfully wrote benchmark data\n";
}

std::string WhisperDiarize::format_timestamp(double seconds) {
    int hours = static_cast<int>(seconds) / 3600;
    int minutes = (static_cast<int>(seconds) % 3600) / 60;
    double secs = fmod(seconds, 60.0);
    
    std::stringstream ss;
    ss << std::setfill('0') << std::setw(2) << hours << ":"
       << std::setfill('0') << std::setw(2) << minutes << ":"
       << std::fixed << std::setprecision(3) << secs;
    return ss.str();
}

// ... implement remaining methods ... 