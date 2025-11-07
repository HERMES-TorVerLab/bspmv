#ifndef BSPMV_UTILS
#define BSPMV_UTILS

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <set>
#include <sstream>
#include <cstdint>
#include "../format/Matrix.hpp"


inline void printBinVal(const uint32_t& value, std::ostream &os) {
    for (int bit = 0; bit < 32; ++bit) {
        os << ((value >> bit) & 1u);
    }
    os << "\n";
}

inline void printVector(const std::vector<uint32_t>& y, const std::string& name = "y") {
    std::cout << name << " = [ ";
    for (size_t i = 0; i < y.size(); ++i) {
        std::cout << y[i];
        if (i + 1 < y.size()) std::cout << ", ";
    }
    std::cout << " ]\n";
}

inline void printVectorBinary32(const std::vector<uint32_t>& y, const std::string& filename = "y.bin32") {
    std::ofstream outFile(filename);
    if (!outFile) {
        std::cerr << "Error: Could not open file " << filename << " for writing.\n";
        return;
    }

    for (size_t i = 0; i < y.size(); ++i) {
        std::bitset<32> bits(y[i]);
        std::string bitString = bits.to_string();  // MSB -> LSB
        std::reverse(bitString.begin(), bitString.end());  // flip to LSB -> MSB
        outFile << bitString;
        if (i + 1 < y.size()) outFile << "\n";
    }

    outFile.close();
}


#endif