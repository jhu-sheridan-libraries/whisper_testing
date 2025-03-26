#include <iostream>
#include <fstream>
#include <string>
#include <vector>

// Function to check if a row is empty or contains only zeros
bool isEmptyOrZeroRow(const std::string& row) {
    // Check if row is empty
    if (row.empty()) {
        return true;
    }
    
    // Count commas and check for non-zero values
    bool hasNonZeroValue = false;
    
    for (char c : row) {
        if (c != ',' && c != '0' && c != ' ' && c != '\r' && c != '\n') {
            hasNonZeroValue = true;
            break;
        }
    }
    
    return !hasNonZeroValue;
}

int main(int argc, char** argv) {
    std::string filename = "whisper_benchmarks.csv";
    
    if (argc > 1) {
        filename = argv[1];
    }
    
    std::cout << "Cleaning up benchmark file: " << filename << std::endl;
    
    // Read the file
    std::ifstream inFile(filename);
    if (!inFile) {
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
    int removedCount = 0;
    while (std::getline(inFile, line)) {
        if (!isEmptyOrZeroRow(line)) {
            validRows.push_back(line);
        } else {
            removedCount++;
        }
    }
    
    inFile.close();
    
    // Write back to file
    std::ofstream outFile(filename);
    if (!outFile) {
        std::cerr << "Error: Could not open file for writing " << filename << std::endl;
        return 1;
    }
    
    for (const auto& row : validRows) {
        outFile << row << std::endl;
    }
    
    outFile.close();
    
    std::cout << "Cleanup complete. Removed " << removedCount << " empty rows. "
              << validRows.size()-1 << " valid entries retained." << std::endl;
    
    return 0;
} 