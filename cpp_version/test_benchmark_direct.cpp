#include <iostream>
#include <fstream>
#include <string>
#include <chrono>
#include <thread>
#include <ctime>
#include <sstream>
#include <iomanip>

std::string getCurrentTimestamp() {
    auto now = std::chrono::system_clock::now();
    auto time_t_now = std::chrono::system_clock::to_time_t(now);
    std::stringstream ss;
    ss << std::put_time(std::localtime(&time_t_now), "%Y-%m-%d %H:%M:%S");
    return ss.str();
}

int main() {
    const std::string csvPath = "direct_benchmark_test.csv";
    
    // Create or overwrite file with header
    {
        std::ofstream file(csvPath);
        if (!file.is_open()) {
            std::cerr << "ERROR: Could not open file for writing: " << csvPath << std::endl;
            return 1;
        }
        
        file << "timestamp,filename,duration_seconds,num_speakers,model,processing_time_seconds,real_time_factor,cpu_percent,cpu_count,memory_percent\n";
        file.close();
    }
    
    // Simulate some processing
    std::cout << "Simulating processing for 2 seconds..." << std::endl;
    auto start = std::chrono::high_resolution_clock::now();
    std::this_thread::sleep_for(std::chrono::seconds(2));
    auto end = std::chrono::high_resolution_clock::now();
    double processingTime = std::chrono::duration<double>(end - start).count();
    
    // Append data
    {
        std::ofstream file(csvPath, std::ios::app);
        if (!file.is_open()) {
            std::cerr << "ERROR: Could not open file for appending: " << csvPath << std::endl;
            return 1;
        }
        
        const std::string filename = "test_file.mp3";
        const double duration = 30.0;
        const int num_speakers = 2;
        const std::string model = "tiny";
        const double rtf = processingTime / duration;
        const double cpu_percent = 50.0;
        const int cpu_count = 8;
        const double memory_percent = 25.0;
        
        file << getCurrentTimestamp() << ","
             << filename << ","
             << duration << ","
             << num_speakers << ","
             << model << ","
             << processingTime << ","
             << rtf << ","
             << cpu_percent << ","
             << cpu_count << ","
             << memory_percent << "\n";
             
        file.close();
    }
    
    // Verify the file was written correctly
    {
        std::ifstream file(csvPath);
        if (!file.is_open()) {
            std::cerr << "ERROR: Could not open file for reading: " << csvPath << std::endl;
            return 1;
        }
        
        std::string line;
        int lineCount = 0;
        
        while (std::getline(file, line)) {
            std::cout << "Line " << ++lineCount << ": " << line << std::endl;
        }
        
        file.close();
    }
    
    std::cout << "Test complete! Check " << csvPath << " for results." << std::endl;
    return 0;
} 