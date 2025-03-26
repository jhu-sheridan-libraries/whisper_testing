#pragma once

#include <iostream>
#include <fstream>
#include <chrono>
#include <string>
#include <ctime>
#include <sstream>
#include <iomanip>

class BenchmarkDebug {
public:
    BenchmarkDebug(const std::string& csvPath = "debug_benchmark.csv") : csvPath(csvPath) {
        std::cout << "BenchmarkDebug initialized with file: " << csvPath << std::endl;
        createCsvIfNotExists();
    }
    
    void start() {
        std::cout << "BenchmarkDebug start() called" << std::endl;
        startTime = std::chrono::high_resolution_clock::now();
    }
    
    void stop(const std::string& filename, double duration, int speakers, const std::string& model) {
        std::cout << "BenchmarkDebug stop() called with:" << std::endl;
        std::cout << "  filename: " << filename << std::endl;
        std::cout << "  duration: " << duration << " seconds" << std::endl;
        std::cout << "  speakers: " << speakers << std::endl;
        std::cout << "  model: " << model << std::endl;
        
        auto endTime = std::chrono::high_resolution_clock::now();
        processingTime = std::chrono::duration<double>(endTime - startTime).count();
        rtf = duration > 0 ? processingTime / duration : 0;
        
        std::cout << "  processingTime: " << processingTime << " seconds" << std::endl;
        std::cout << "  rtf: " << rtf << std::endl;
        
        appendResult(filename, duration, speakers, model);
    }
    
    double getProcessingTime() const { 
        std::cout << "BenchmarkDebug getProcessingTime() called, returning " << processingTime << std::endl;
        return processingTime; 
    }
    
    double getRealTimeFactor() const { 
        std::cout << "BenchmarkDebug getRealTimeFactor() called, returning " << rtf << std::endl;
        return rtf; 
    }

private:
    std::string csvPath;
    std::chrono::time_point<std::chrono::high_resolution_clock> startTime;
    double processingTime = 0.0;
    double rtf = 0.0;

    void createCsvIfNotExists() {
        std::ifstream checkFile(csvPath);
        if (!checkFile.is_open()) {
            std::cout << "Creating new CSV file: " << csvPath << std::endl;
            std::ofstream file(csvPath);
            if (file.is_open()) {
                file << "timestamp,filename,duration_seconds,num_speakers,model,processing_time_seconds,real_time_factor,cpu_percent,cpu_count,memory_percent\n";
                file.close();
                std::cout << "CSV file created successfully" << std::endl;
            } else {
                std::cerr << "ERROR: Could not create CSV file: " << csvPath << std::endl;
            }
        } else {
            std::cout << "CSV file already exists" << std::endl;
            checkFile.close();
        }
    }

    std::string getCurrentTimestamp() {
        auto now = std::chrono::system_clock::now();
        auto time_t_now = std::chrono::system_clock::to_time_t(now);
        std::stringstream ss;
        ss << std::put_time(std::localtime(&time_t_now), "%Y-%m-%d %H:%M:%S");
        return ss.str();
    }

    void appendResult(const std::string& filename, double duration, int speakers, const std::string& model) {
        std::cout << "BenchmarkDebug appendResult() called" << std::endl;
        
        // First try to read the file to check if it exists and has content
        {
            std::ifstream checkFile(csvPath);
            std::cout << "Checking if file can be opened for reading: " << (checkFile.is_open() ? "Yes" : "No") << std::endl;
            
            if (checkFile.is_open()) {
                std::string line;
                int lineCount = 0;
                while (std::getline(checkFile, line) && lineCount < 5) {
                    std::cout << "Existing line " << ++lineCount << ": " << line << std::endl;
                }
                checkFile.close();
            }
        }
        
        // Now try to append
        std::ofstream file(csvPath, std::ios::app);
        if (file.is_open()) {
            std::cout << "File opened successfully for appending" << std::endl;
            
            std::string timestamp = getCurrentTimestamp();
            std::cout << "Generated timestamp: " << timestamp << std::endl;
            
            // For demo purposes, use fixed values for system metrics
            double cpuPercent = 40.0;
            int cpuCount = 8;
            double memoryPercent = 30.0;
            
            file << timestamp << ","
                 << filename << ","
                 << duration << ","
                 << speakers << ","
                 << model << ","
                 << processingTime << ","
                 << rtf << ","
                 << cpuPercent << ","
                 << cpuCount << ","
                 << memoryPercent << "\n";
                 
            file.close();
            std::cout << "Data written successfully to " << csvPath << std::endl;
        } else {
            std::cerr << "ERROR: Could not open file for appending: " << csvPath << std::endl;
        }
        
        // Verify what was written
        {
            std::ifstream verifyFile(csvPath);
            if (verifyFile.is_open()) {
                std::string line;
                int lineCount = 0;
                std::cout << "File contents after write:" << std::endl;
                while (std::getline(verifyFile, line)) {
                    std::cout << "Line " << ++lineCount << ": " << line << std::endl;
                }
                verifyFile.close();
            } else {
                std::cerr << "ERROR: Could not open file for verification: " << csvPath << std::endl;
            }
        }
    }
}; 