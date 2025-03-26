#include "benchmark.h"
#include <iostream>
#include <cstdio>
#include <cstring>
#include <sys/resource.h>
#include <thread>
#include <fstream>
#include <sstream>
#include <vector>
#include <algorithm>

#ifdef __linux__
#include <sys/sysinfo.h>
#include <unistd.h>
#elif defined(__APPLE__)
#include <mach/mach.h>
#include <sys/sysctl.h>
#endif

Benchmark::Benchmark(const std::string& csv_path) : csv_path(csv_path) {
    createCsvIfNotExists();
}

void Benchmark::start() {
    start_time = std::chrono::high_resolution_clock::now();
}

void Benchmark::stop(const std::string& filename, 
                    double duration_seconds, 
                    int num_speakers, 
                    const std::string& model) {
    auto end_time = std::chrono::high_resolution_clock::now();
    processing_time_seconds = std::chrono::duration<double>(end_time - start_time).count();
    
    // Calculate real-time factor (processing time / audio duration)
    real_time_factor = duration_seconds > 0 ? processing_time_seconds / duration_seconds : 0;
    
    // Get system metrics
    updateSystemMetrics();
    
    // Record the benchmark
    appendResult(filename, duration_seconds, num_speakers, model);
}

bool Benchmark::fileExists(const std::string& path) {
    std::ifstream file(path);
    return file.good();
}

void Benchmark::createCsvIfNotExists() {
    if (!fileExists(csv_path)) {
        std::ofstream file(csv_path);
        if (file.is_open()) {
            file << "timestamp,filename,duration_seconds,num_speakers,model,processing_time_seconds,real_time_factor,cpu_percent,cpu_count,memory_percent\n";
            file.close();
        } else {
            std::cerr << "Error: Could not create benchmark file at " << csv_path << std::endl;
        }
    }
}

// Function to clean up the benchmark file by removing empty rows
bool cleanupBenchmarkFile(const std::string& filepath) {
    // Check if file exists
    std::ifstream checkFile(filepath);
    if (!checkFile.is_open()) {
        return false;  // File doesn't exist, nothing to clean
    }
    checkFile.close();

    // Read the file content
    std::ifstream inFile(filepath);
    if (!inFile.is_open()) {
        std::cerr << "Cannot open file for cleanup: " << filepath << std::endl;
        return false;
    }

    // Read all rows from the file
    std::vector<std::string> rows;
    std::string line;
    
    // First line is the header
    if (std::getline(inFile, line)) {
        rows.push_back(line);
    }
    
    // Read all other lines
    while (std::getline(inFile, line)) {
        rows.push_back(line);
    }
    
    inFile.close();
    
    // Filter out empty rows
    std::vector<std::string> validRows;
    validRows.push_back(rows[0]);  // Keep header
    
    for (size_t i = 1; i < rows.size(); i++) {
        const std::string& row = rows[i];
        
        // Skip if row is empty
        if (row.empty()) {
            continue;
        }
        
        // Count non-zero, non-comma characters
        int nonZeroCount = 0;
        for (char c : row) {
            if (c != ',' && c != '0' && c != ' ' && c != '\r' && c != '\n') {
                nonZeroCount++;
            }
        }
        
        // Only add row if it has some non-zero content
        if (nonZeroCount > 0) {
            validRows.push_back(row);
        }
    }
    
    // Write back only valid rows
    std::ofstream outFile(filepath);
    if (!outFile.is_open()) {
        std::cerr << "Cannot open file for writing: " << filepath << std::endl;
        return false;
    }
    
    for (const auto& row : validRows) {
        outFile << row << std::endl;
    }
    
    outFile.close();
    return true;
}

void Benchmark::appendResult(const std::string& filename, 
                           double duration_seconds, 
                           int num_speakers, 
                           const std::string& model) {
    std::cout << "DEBUG: Writing benchmark data:" << std::endl;
    std::cout << "  - Filename: " << filename << std::endl;
    std::cout << "  - Duration: " << duration_seconds << " seconds" << std::endl;
    std::cout << "  - Speakers: " << num_speakers << std::endl;
    std::cout << "  - Model: " << model << std::endl;
    std::cout << "  - Processing time: " << processing_time_seconds << " seconds" << std::endl;
    std::cout << "  - RTF: " << real_time_factor << std::endl;
    
    // Open file in append mode
    std::ofstream file(csv_path, std::ios::app);
    if (file.is_open()) {
        file << getCurrentTimestamp() << ","
             << filename << ","
             << duration_seconds << ","
             << num_speakers << ","
             << model << ","
             << processing_time_seconds << ","
             << real_time_factor << ","
             << cpu_percent << ","
             << cpu_count << ","
             << memory_percent << "\n";
        file.close();
        std::cout << "DEBUG: Successfully wrote to " << csv_path << std::endl;
    } else {
        std::cerr << "Error: Could not open benchmark file for writing at " << csv_path << std::endl;
    }
}

void Benchmark::updateSystemMetrics() {
    cpu_count = getCpuCores();
    cpu_percent = getCurrentCpuUsage();
    memory_percent = getCurrentMemoryUsage();
}

int Benchmark::getCpuCores() {
    return std::thread::hardware_concurrency();
}

double Benchmark::getCurrentCpuUsage() {
    // This is a simplified implementation
    // For more accurate CPU usage, you would need to sample multiple times
#ifdef __linux__
    // On Linux, read from /proc/stat
    FILE* file = fopen("/proc/stat", "r");
    if (file) {
        long user, nice, system, idle, iowait, irq, softirq;
        char cpu[10];
        fscanf(file, "%s %ld %ld %ld %ld %ld %ld %ld", 
               cpu, &user, &nice, &system, &idle, &iowait, &irq, &softirq);
        fclose(file);
        
        long total_idle = idle + iowait;
        long total = user + nice + system + idle + iowait + irq + softirq;
        
        // Simple approximation - in a real implementation, you'd compare with previous values
        return 100.0 * (1.0 - (double)total_idle / total);
    }
#endif
    // Default fallback
    return 0.0;
}

double Benchmark::getCurrentMemoryUsage() {
#ifdef __linux__
    struct sysinfo memInfo;
    sysinfo(&memInfo);
    
    long long totalPhysMem = memInfo.totalram;
    totalPhysMem *= memInfo.mem_unit;
    
    long long physMemUsed = memInfo.totalram - memInfo.freeram;
    physMemUsed *= memInfo.mem_unit;
    
    return 100.0 * ((double)physMemUsed / totalPhysMem);
#elif defined(__APPLE__)
    // macOS implementation would go here
    vm_size_t page_size;
    mach_port_t mach_port = mach_host_self();
    vm_statistics64_data_t vm_stats;
    mach_msg_type_number_t count = sizeof(vm_stats) / sizeof(natural_t);
    
    if (host_page_size(mach_port, &page_size) == KERN_SUCCESS &&
        host_statistics64(mach_port, HOST_VM_INFO64, (host_info64_t)&vm_stats, &count) == KERN_SUCCESS) {
        
        long long free_memory = (int64_t)vm_stats.free_count * (int64_t)page_size;
        long long used_memory = ((int64_t)vm_stats.active_count + 
                                (int64_t)vm_stats.inactive_count + 
                                (int64_t)vm_stats.wire_count) * (int64_t)page_size;
        long long total_memory = free_memory + used_memory;
        
        return 100.0 * ((double)used_memory / total_memory);
    }
#endif
    // Default fallback
    return 0.0;
}

std::string Benchmark::getCurrentTimestamp() {
    auto now = std::chrono::system_clock::now();
    auto time_t_now = std::chrono::system_clock::to_time_t(now);
    
    std::stringstream ss;
    ss << std::put_time(std::localtime(&time_t_now), "%Y-%m-%d %H:%M:%S");
    return ss.str();
} 