#include "benchmark.h"
#include <iostream>
#include <thread>
#include <chrono>

int main() {
    // Create benchmark object
    Benchmark benchmark;
    
    std::cout << "Starting benchmark test..." << std::endl;
    
    // Start benchmarking
    benchmark.start();
    
    // Simulate processing time (sleep for 2 seconds)
    std::cout << "Simulating processing for 2 seconds..." << std::endl;
    std::this_thread::sleep_for(std::chrono::seconds(2));
    
    // Stop benchmarking with actual values
    const char* test_file = "test_audio.mp3";
    const double audio_duration = 30.0;  // 30 second audio file
    const int num_speakers = 2;
    const char* model = "tiny";
    
    benchmark.stop(test_file, audio_duration, num_speakers, model);
    
    std::cout << "Benchmark test complete!" << std::endl;
    std::cout << "Check whisper_benchmarks.csv for the new entry." << std::endl;
    
    return 0;
} 