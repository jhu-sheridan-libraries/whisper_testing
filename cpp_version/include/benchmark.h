#ifndef BENCHMARK_H
#define BENCHMARK_H

#include <string>
#include <chrono>
#include <fstream>
#include <vector>
#include <ctime>
#include <iomanip>
#include <sstream>

class Benchmark {
public:
    Benchmark(const std::string& csv_path = "whisper_benchmarks.csv");
    ~Benchmark() = default;

    void start();
    void stop(const std::string& filename, 
              double duration_seconds, 
              int num_speakers, 
              const std::string& model);
    
    // Getters for metrics
    double getProcessingTime() const { return processing_time_seconds; }
    double getRealTimeFactor() const { return real_time_factor; }
    double getCpuPercent() const { return cpu_percent; }
    int getCpuCount() const { return cpu_count; }
    double getMemoryPercent() const { return memory_percent; }

private:
    std::string csv_path;
    std::chrono::time_point<std::chrono::high_resolution_clock> start_time;
    double processing_time_seconds = 0.0;
    double real_time_factor = 0.0;
    double cpu_percent = 0.0;
    int cpu_count = 0;
    double memory_percent = 0.0;

    bool fileExists(const std::string& path);
    void createCsvIfNotExists();
    void appendResult(const std::string& filename, 
                     double duration_seconds, 
                     int num_speakers, 
                     const std::string& model);
    
    // System monitoring functions
    void updateSystemMetrics();
    int getCpuCores();
    double getCurrentCpuUsage();
    double getCurrentMemoryUsage();
    std::string getCurrentTimestamp();
};

#endif // BENCHMARK_H 