#include "Matrix.hpp"


MatrixBwcCsr::MatrixBwcCsr(){
    this->format = MatrixFormat::BWC_CSR;
}

void MatrixBwcCsr::print(const std::string& file_name = "mtx_bwc_csr.txt") const {
    if (rowPtr.size() != rows + 1) {
        throw std::runtime_error("rowPtr size must be rows + 1");
    }
    if (colIdx.size() != word.size()) {
        throw std::runtime_error("colIdx and word must have the same length");
    }

    std::ofstream file(file_name);
    if (!file.is_open()) {
        throw std::runtime_error("Error: Could not open file " + file_name);
    }

    file << "Word: " << word.size() << "\n";

    for (uint32_t i = 0; i < rows; ++i) {
        uint32_t row_start = rowPtr[i];
        uint32_t row_end   = rowPtr[i + 1];

        file << "ROW " << i << "\n";

        for (uint32_t j = row_start; j < row_end; ++j) {
            file << "(row=" << i
                 << ", wordCol=" << colIdx[j]
                 << ") ";
            printBinVal(word[j], file);
        }
    }
}
