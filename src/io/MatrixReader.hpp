/**
 * @file MatrixReader.hpp
 * @author Stack1
 * @brief 
 * @version 1.0
 * @date 19-09-2025
 * 
 * 
 */
#pragma once

class MatrixBase;          // forward declaration
enum class MatrixFormat;

struct MatrixMarketHeader {
    uint32_t rows = 0;
    uint32_t cols = 0;
    uint32_t nnz  = 0;
    bool isSymmetric = false;
    bool isBinary    = false;
};


namespace MatrixReader {
    MatrixBase* readMatrixFromFile(const std::string& filename, MatrixFormat format);
}

