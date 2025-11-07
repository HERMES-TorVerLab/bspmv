#include "Matrix.hpp"

// ===== MatrixCSR =====

MatrixCSR::MatrixCSR(){
    this->format = MatrixFormat::CSR;
}


void MatrixCSR::print(const std::string& file_name = "mtx_csr.txt") const {
    std::ofstream file(file_name);

    if (!file.is_open()) {
        std::cerr << "Error: Could not open file.\n";
        std::exit(EXIT_FAILURE);
    }

    for (uint32_t i = 0; i < rows; ++i) {
        uint32_t row_start = rowPtr[i];
        uint32_t row_end = rowPtr[i + 1];

        file << "ROW: " << i + 1 << "\n" << (row_end - row_start) << ": ";

        for (uint32_t j = row_start; j < row_end; ++j) {
            file << colIdx[j] << " ";
        }
        file << "\n";
    }

    file.close();
}


