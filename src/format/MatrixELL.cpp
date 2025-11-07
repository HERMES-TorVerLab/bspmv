#include "Matrix.hpp"


// ===== MatrixELL =====

MatrixELL::MatrixELL() {
    this->format = MatrixFormat::ELL;
}

void MatrixELL::print(const std::string& file_name = "mtx_ell.txt") const {
    std::ofstream file(file_name);

    if (!file.is_open()) {
        std::cerr << "Error: Could not open file.\n";
        std::exit(EXIT_FAILURE);
    }

    for (uint32_t i = 0; i < k; ++i) {
        file << "\nROW " << i + 1 << ": ";
        
        for(uint32_t j = 0; j < rows; ++j){
            file << colIdx[j + i * rows] << " ";
        }
        file << "\n";

    }

    file.close();
}
