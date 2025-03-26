#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <sstream>

void showHelp() {
    std::cout << "Benchmark File Manager\n";
    std::cout << "Usage:\n";
    std::cout << "  manage_benchmarks [command]\n\n";
    std::cout << "Commands:\n";
    std::cout << "  reset    - Reset the benchmark file to a clean state\n";
    std::cout << "  clean    - Remove empty rows from the benchmark file\n";
    std::cout << "  show     - Display the current benchmark file contents\n";
    std::cout << "  help     - Show this help message\n";
}

void resetBenchmarkFile() {
    std::ofstream file("whisper_benchmarks.csv");
    if (!file.is_open()) {
        std::cerr << "Error: Could not open benchmark file for writing\n";
        return;
    }
    
    file << "timestamp,filename,duration_seconds,num_speakers,model,processing_time_seconds,real_time_factor,cpu_percent,cpu_count,memory_percent\n";
    file << "2025-03-25 00:10:24,SFd2CK2qiVI.mp3,1300.885,1,base,8814.24968290329,6.7755794577562884,5.3,8,72.9\n";
    file.close();
    
    std::cout << "Benchmark file has been reset to a clean state.\n";
}

void cleanBenchmarkFile() {
    std::ifstream inFile("whisper_benchmarks.csv");
    if (!inFile.is_open()) {
        std::cerr << "Error: Could not open benchmark file for reading\n";
        return;
    }
    
    std::vector<std::string> validRows;
    std::string line;
    
    // Read header
    if (std::getline(inFile, line)) {
        validRows.push_back(line);
    }
    
    // Read data rows
    while (std::getline(inFile, line)) {
        // Skip empty or zero-only rows
        bool isEmpty = true;
        for (char c : line) {
            if (c != ',' && c != '0' && c != ' ' && c != '\r' && c != '\n') {
                isEmpty = false;
                break;
            }
        }
        
        if (!isEmpty && !line.empty()) {
            validRows.push_back(line);
        }
    }
    
    inFile.close();
    
    // Write back clean data
    std::ofstream outFile("whisper_benchmarks.csv");
    if (!outFile.is_open()) {
        std::cerr << "Error: Could not open benchmark file for writing\n";
        return;
    }
    
    for (const auto& row : validRows) {
        outFile << row << "\n";
    }
    
    outFile.close();
    
    std::cout << "Benchmark file has been cleaned of empty rows.\n";
}

void showBenchmarkFile() {
    std::ifstream file("whisper_benchmarks.csv");
    if (!file.is_open()) {
        std::cerr << "Error: Could not open benchmark file for reading\n";
        return;
    }
    
    std::string line;
    while (std::getline(file, line)) {
        std::cout << line << "\n";
    }
    
    file.close();
}

int main(int argc, char** argv) {
    if (argc < 2) {
        showHelp();
        return 0;
    }
    
    std::string command = argv[1];
    
    if (command == "reset") {
        resetBenchmarkFile();
    } else if (command == "clean") {
        cleanBenchmarkFile();
    } else if (command == "show") {
        showBenchmarkFile();
    } else if (command == "help") {
        showHelp();
    } else {
        std::cerr << "Unknown command: " << command << "\n";
        showHelp();
        return 1;
    }
    
    return 0;
} 