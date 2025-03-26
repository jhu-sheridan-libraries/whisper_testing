#include "whisper_diarize.h"
#include <iostream>
#include <string>
#include <cxxopts.hpp>
#include <iomanip>

int main(int argc, char** argv) {
    try {
        cxxopts::Options options("whisper_diarize", "Audio transcription with speaker diarization");
        
        options.add_options()
            ("a,audio", "Audio file path", cxxopts::value<std::string>())
            ("o,output", "Output file path", cxxopts::value<std::string>())
            ("m,model", "Model size (tiny/base/small/medium/large)", cxxopts::value<std::string>()->default_value("medium"))
            ("f,format", "Output format (vtt/txt)", cxxopts::value<std::string>()->default_value("vtt"))
            ("s,speakers", "Number of speakers", cxxopts::value<int>()->default_value("0"))
            ("l,language", "Language code", cxxopts::value<std::string>()->default_value(""))
            ("h,help", "Print usage");
            
        auto result = options.parse(argc, argv);
        
        if (result.count("help") || !result.count("audio")) {
            std::cout << options.help() << std::endl;
            return 0;
        }
        
        // Get parameters
        std::string audio_path = result["audio"].as<std::string>();
        std::string output_path = result["output"].as<std::string>();
        std::string model_name = result["model"].as<std::string>();
        std::string model_path = "models/ggml-" + model_name + ".bin";
        std::string format_str = result["format"].as<std::string>();
        int num_speakers = result["speakers"].as<int>();
        std::string language = result["language"].as<std::string>();
        
        // Set up options
        TranscriptionOptions transcription_options;
        transcription_options.model_path = model_path;
        transcription_options.language = language;
        transcription_options.num_speakers = num_speakers;
        
        if (format_str == "vtt") {
            transcription_options.format = OutputFormat::VTT;
        } else if (format_str == "txt") {
            transcription_options.format = OutputFormat::TXT;
        } else {
            std::cerr << "Unknown format: " << format_str << std::endl;
            return 1;
        }
        
        // Initialize and run transcription
        WhisperDiarize diarizer;
        if (!diarizer.initialize(transcription_options)) {
            std::cerr << "Failed to initialize WhisperDiarize" << std::endl;
            return 1;
        }
        
        bool success = diarizer.transcribe(audio_path, output_path);
        
        if (success) {
            std::cout << "\nTranscription completed successfully!" << std::endl;
            std::cout << "Output written to: " << output_path << std::endl;
            
            // Display benchmark results
            std::cout << "\nPerformance metrics:" << std::endl;
            std::cout << "  Processing time: " << std::fixed << std::setprecision(2) 
                      << diarizer.getProcessingTime() << " seconds" << std::endl;
            std::cout << "  Real-time factor: " << std::fixed << std::setprecision(2) 
                      << diarizer.getRealTimeFactor() << "x" << std::endl;
            std::cout << "  (Results saved to whisper_benchmarks.csv)" << std::endl;
        } else {
            std::cerr << "Transcription failed." << std::endl;
            return 1;
        }
        
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
} 