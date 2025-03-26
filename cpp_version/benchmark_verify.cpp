#include <iostream>
#include <fstream>
#include <string>
#include <chrono>
#include <thread>

int main() {
    // Clear out existing file
    std::ofstream clear("whisper_benchmarks.csv");
    if (clear.is_open()) {
        clear << "timestamp,filename,duration_seconds,num_speakers,model,processing_time_seconds,real_time_factor,cpu_percent,cpu_count,memory_percent\n";
        clear.close();
        std::cout << "Created clean benchmark file" << std::endl;
    } else {
        std::cerr << "ERROR: Could not open benchmark file for writing" << std::endl;
        return 1;
    }

    // Open file for appending
    std::ofstream file("whisper_benchmarks.csv", std::ios::app);
    if (!file.is_open()) {
        std::cerr << "ERROR: Could not open benchmark file for appending" << std::endl;
        return 1;
    }

    // Get current timestamp
    auto now = std::chrono::system_clock::now();
    auto time_t_now = std::chrono::system_clock::to_time_t(now);
    char timestamp[64];
    strftime(timestamp, sizeof(timestamp), "%Y-%m-%d %H:%M:%S", localtime(&time_t_now));

    // Write a sample record
    file << timestamp << ","
         << "benchmark_test.mp3" << ","
         << "60.0" << ","
         << "2" << ","
         << "tiny" << ","
         << "3.0" << ","
         << "0.05" << ","
         << "40.0" << ","
         << "8" << ","
         << "25.0" << "\n";

    file.close();
    std::cout << "Added sample benchmark record" << std::endl;

    // Verify what was written
    std::ifstream verify("whisper_benchmarks.csv");
    if (verify.is_open()) {
        std::string line;
        while (std::getline(verify, line)) {
            std::cout << line << std::endl;
        }
        verify.close();
    }

    std::cout << "Benchmark verification complete!" << std::endl;
    return 0;
} 