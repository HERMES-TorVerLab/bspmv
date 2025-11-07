#include "Matrix.hpp"


// ===== MatrixHLL =====

MatrixHLL::MatrixHLL(){
    this->format = MatrixFormat::HLL;
}

void MatrixHLL::print(const std::string& file_name = "mtx_hll_col_major.txt") const {
    std::ofstream file(file_name);

    if (!file.is_open()) {
        std::cerr << "Error: Could not open file.\n";
        std::exit(EXIT_FAILURE);
    }

    uint32_t numHacks = hack.size() - 1;

    for (uint32_t h = 0; h < numHacks; ++h) {
        uint32_t hackStart = hack[h];
        uint32_t hackEnd = hack[h + 1];

        // Number of actual rows in this hack (could be < hackSize at the end)
        uint32_t localRows = std::min(hackSize, rows - h * hackSize);
        uint32_t hackWidth = (hackEnd - hackStart) / localRows;

        file << "\nHACK " << h + 1 
             << " (Rows " << h * hackSize + 1 << " to " << h * hackSize + localRows 
             << ", Width: " << hackWidth << "):\n";

        for (uint32_t c = 0; c < hackWidth; ++c) {
            file << "ROW " << c + 1 << ": ";
            for (uint32_t r = 0; r < localRows; ++r) {
                uint32_t idx = hackStart + c * localRows + r;
                file << colIdx[idx] << " ";
            }
            file << "\n";
        }
    }

    file.close();
}
