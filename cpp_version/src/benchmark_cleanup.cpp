#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <sstream>

// Function to check if a row is empty or contains only zeros
bool isEmptyOrZeroRow(const std::string& row) {
    // Check if row is empty
    if (row.empty() || row == "\n" || row == "\r\n") {
        return true;
    }
    
    // Count commas and check for non-zero values
    int commaCount = 0;
    bool hasNonZeroValue = false;
    
    for (size_t i = 0; i < row.length(); i++) {
        if (row[i] == ',') {
            commaCount++;
        } else if (row[i] != '0' && row[i] != ' ' && row[i] != '\n' && row[i] != '\r') {
            hasNonZeroValue = true;
        }
    }
    
    // If it has 9 commas (10 fields) and no non-zero values, it's likely an empty row
    return (commaCount == 9 && !hasNonZeroValue);
}

int main(int argc, char** argv) {
    std::string filename = "whisper_benchmarks.csv";
    
    if (argc > 1) {
        filename = argv[1];
    }
    
    std::cout << "Cleaning up benchmark file: " << filename << std::endl;
    
    // Read the file
    std::ifstream inFile(filename);
    if (!inFile.is_open()) {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        return 1;
    }
    
    std::vector<std::string> validRows;
    std::string line;
    
    // Read header
    if (std::getline(inFile, line)) {
        validRows.push_back(line);
    }
    
    // Read data rows
    while (std::getline(inFile, line)) {
        if (!isEmptyOrZeroRow(line)) {
            validRows.push_back(line);
        }
    }
    
    inFile.close();
    
    // Write back to file
    std::ofstream outFile(filename);
    if (!outFile.is_open()) {
        std::cerr << "Error: Could not open file for writing " << filename << std::endl;
        return 1;
    }
    
    for (const auto& row : validRows) {
        outFile << row << std::endl;
    }
    
    outFile.close();
    
    std::cout << "Cleanup complete. Removed " 
              << (validRows.size() > 0 ? validRows.size() - 1 : 0) 
              << " valid entries retained." << std::endl;
    
    return 0;
} 