#include "Matrix.hpp"


MatrixCOO::MatrixCOO(){
    this->format = MatrixFormat::COO;
}

void MatrixCOO::print(const std::string& file_name = "mtx_coo.txt") const {
    std::ofstream file(file_name);
    uint32_t lastRow = 1U;

    if (!file.is_open()) {
        std::cerr << "Error: Could not open file.\n";
        std::exit(EXIT_FAILURE);
    }

    file << "ROW: 1\n"; 
    for (uint32_t i = 0; i < nnz; ++i) {
        if(rowIdx[i] > lastRow){
            file << "\n";
            file << "ROW: " << rowIdx[i] << "\n";
            lastRow = rowIdx[i];
        }
        file << "(" << rowIdx[i] << ", " << colIdx[i] << ") ";
    }
    file << "\n";

    file.close();
}