#include "Matrix.hpp"


MatrixBwcCoo::MatrixBwcCoo(){
    this->format = MatrixFormat::BWC_COO;
}

void MatrixBwcCoo::print(const std::string& file_name = "mtx_bwc_coo.txt") const {
    if (rowIdx.size() != word.size() || colIdx.size() != word.size() || numWord != word.size()) {
        throw std::runtime_error("rowIdx, colIdx, and word must all have the same length");
    }

    std::ofstream file(file_name);
    if (!file.is_open()) {
        throw std::runtime_error("Error: Could not open file " + file_name);
    }

    file << "Word: " << numWord << "\n";

    uint32_t currRow = rowIdx.empty() ? 0U : rowIdx[0];
    file << "ROW " << currRow << "\n";

    for (size_t i = 0; i < word.size(); ++i) {
        if (rowIdx[i] > currRow) {
            currRow = rowIdx[i];
            file << "ROW " << currRow << "\n";
        }
        file << "(row=" << rowIdx[i]
             << ", wordCol=" << colIdx[i]
             << ") ";
        printBinVal(word[i], file);
    }
}
