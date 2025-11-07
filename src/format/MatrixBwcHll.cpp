#include "Matrix.hpp"


MatrixBwcHll::MatrixBwcHll(){
    this->format = MatrixFormat::BWC_HLL;
}



void MatrixBwcHll::print(const std::string& file_name = "mtx_bwc_hll.txt") const {
    std::ofstream file(file_name);

    if (!file.is_open()) {
        throw std::runtime_error("Error: Could not open file " + file_name);
    }

    file << "MatrixBWC-HLL with " << rows << " rows, hackSize = " << hackSize << "\n";

    uint32_t numHacks = hack.size() - 1;

    for (uint32_t h = 0; h < numHacks; ++h) {
        uint32_t offset = hack[h];
        uint32_t nextOffset = hack[h + 1];
        uint32_t localRows = std::min(hackSize, rows - h * hackSize);
        uint32_t hackWidth = (nextOffset - offset) / localRows;

        file << "\n[HACK " << h << "] rows " << (h * hackSize) << " to " << (h * hackSize + localRows - 1) << "\n";

        for (uint32_t w = 0; w < hackWidth; ++w) {
            file << "  ROW " << w << ":\n";

            for (uint32_t r = 0; r < localRows; ++r) {
                uint32_t globalRow = h * hackSize + r;
                uint32_t index = offset + w * localRows + r;
                uint32_t col = colIdx[index];

                file << "    (row=" << globalRow << ", wordCol=" << col << ") ";
                printBinVal(word[index], file);
            }
        }
    }

    file.close();
}
