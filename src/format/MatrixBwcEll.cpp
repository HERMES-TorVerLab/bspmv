#include "Matrix.hpp"


MatrixBwcEll::MatrixBwcEll(){
    this->format = MatrixFormat::BWC_ELL;
}

void MatrixBwcEll::print(const std::string& file_name = "mtx_bwc_ell.txt") const {
    std::ofstream file(file_name);

    if (!file.is_open()) {
        throw std::runtime_error("Error: Could not open file " + file_name);
    }

    file << "MatrixELL with " << rows << " rows and max " << k << " entries per row\n";

    for (uint32_t i = 0; i < k; ++i) {
        file << "ROW " << i + 1 << "\n";

        for (uint32_t j = 0; j < rows; ++j) {
            uint32_t col = colIdx[j + i * rows];
            file << "(row=" << i + 1
                 << ", col=" << col
                 << ") ";
            printBinVal(word[j + i * rows], file);
        }
    }

    file.close();
}