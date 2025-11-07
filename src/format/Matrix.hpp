#ifndef MATRIX_HPP
#define MATRIX_HPP

#include <memory>
#include <vector>
#include <iostream>
#include <cstdint>
#include <fstream>   
#include <sstream>   
#include <stdexcept> 
#include <filesystem>
#include <algorithm>
#include <tuple>
#include <cmath>
#include <set>
#include <bitset>
#include "../utils/bspmvUtils.hpp"  

namespace fs = std::filesystem;


// Enum for matrix format types
enum class MatrixFormat {
    COO,
    CSR,
    ELL,
    HLL,
    BWC_COO,
    BWC_CSR,
    BWC_ELL,
    BWC_HLL
};

enum class ExecutionMode { 
    CPU, GPU 
};

// ===== Abstract Base Class =====
class MatrixBase {
    public:
        MatrixFormat format;
        std::uint32_t rows, cols, nnz;

        virtual ~MatrixBase();  // There is no constructor, only coopnstructors of derived classes can be instantiated

        virtual void print(const std::string& fileName = "mtx.txt") const = 0;
        
        MatrixFormat getFormat() const;
        uint32_t getRows() const;
        uint32_t getCols() const;
        uint32_t getNonZeros() const;

        // New virtual getters for word-level stats
        virtual uint32_t getNumWord() const { return 0; }
        virtual uint32_t getMinNnzWord() const { return 0; }
        virtual uint32_t getMaxNnzWord() const { return 0; }
        virtual double   getAvgNnzWord() const { return 0.0; }
        virtual double memoryMB() const = 0; // memory in MB
};

// Helper function to compute vector memory
template<typename T>
size_t vectorMemory(const std::vector<T>& v) {
    return v.size() * sizeof(T);
}


// ===== Derived Matrix Formats =====

class MatrixCOO : public MatrixBase {
    public:
        std::vector<uint32_t> rowIdx;
        std::vector<uint32_t> colIdx;
        bool isRowMajor;
        
        MatrixCOO();

        void print(const std::string& fileName) const override;
        double memoryMB() const override {
            size_t memBytes = vectorMemory(rowIdx) + vectorMemory(colIdx) + sizeof(isRowMajor) + sizeof(MatrixFormat) 
            + sizeof(rows) + sizeof(cols) + sizeof(nnz);
            return static_cast<double>(memBytes) / (1024.0 * 1024.0);
        }
};


class MatrixCSR : public MatrixBase {    
    public:
        std::vector<uint32_t> rowPtr;
        std::vector<uint32_t> colIdx;

        MatrixCSR();

        void print(const std::string& fileName) const override;
        double memoryMB() const override {
            size_t memBytes = vectorMemory(rowPtr) + vectorMemory(colIdx) + sizeof(MatrixFormat) 
            + sizeof(rows) + sizeof(cols) + sizeof(nnz);
            return static_cast<double>(memBytes) / (1024.0 * 1024.0);
        }
};

class MatrixELL : public MatrixBase {
    public:
        uint32_t k;
        std::vector<uint32_t> colIdx;

        MatrixELL();
        void print(const std::string& fileName) const override;
        double memoryMB() const override {
            size_t memBytes = sizeof(k) + vectorMemory(colIdx) + sizeof(MatrixFormat) 
            + sizeof(rows) + sizeof(cols) + sizeof(nnz);
            return static_cast<double>(memBytes) / (1024.0 * 1024.0);
        }
};

class MatrixHLL : public MatrixBase {
    public:
        uint32_t hackSize;
        std::vector<uint32_t> hack;
        std::vector<uint32_t> colIdx;

        MatrixHLL();
        void print(const std::string& fileName) const override;
        double memoryMB() const override {
            size_t memBytes = sizeof(hackSize) + vectorMemory(colIdx) + vectorMemory(hack) 
                            + sizeof(MatrixFormat) 
                            + sizeof(rows) 
                            + sizeof(cols) 
                            + sizeof(nnz);
            return static_cast<double>(memBytes) / (1024.0 * 1024.0);
        }
};


class MatrixBwcCoo : public MatrixBase {
    public:
        uint32_t numWord, minNnzWord, maxNnzWord;
        double avgNnzWord;
        std::vector<uint32_t> rowIdx;
        std::vector<uint32_t> colIdx;
        std::vector<uint32_t> word;

        MatrixBwcCoo();
        void print(const std::string& fileName) const override;
        uint32_t getNumWord() const override { return numWord; }
        uint32_t getMinNnzWord() const override { return minNnzWord; }
        uint32_t getMaxNnzWord() const override { return maxNnzWord; }
        double   getAvgNnzWord() const override { return avgNnzWord; }
        double memoryMB() const override {
            size_t memBytes = sizeof(numWord) + sizeof(minNnzWord) + sizeof(maxNnzWord) + sizeof(avgNnzWord)
                            + vectorMemory(rowIdx)
                            + vectorMemory(colIdx)
                            + vectorMemory(word) 
                            + sizeof(MatrixFormat) + sizeof(rows) + sizeof(cols) + sizeof(nnz);
            return static_cast<double>(memBytes) / (1024.0 * 1024.0);
        }
};

class MatrixBwcCsr : public MatrixBase {
    public:
        uint32_t numWord, minNnzWord, maxNnzWord;
        double avgNnzWord;
        std::vector<uint32_t> rowPtr;
        std::vector<uint32_t> colIdx;
        std::vector<uint32_t> word;
        
        MatrixBwcCsr();
        void print(const std::string& fileName) const override;
        uint32_t getNumWord() const override { return numWord; }
        uint32_t getMinNnzWord() const override { return minNnzWord; }
        uint32_t getMaxNnzWord() const override { return maxNnzWord; }
        double   getAvgNnzWord() const override { return avgNnzWord; }

        double memoryMB() const override {
            size_t memBytes = sizeof(numWord) + sizeof(minNnzWord) + sizeof(maxNnzWord) + sizeof(avgNnzWord)
                            + vectorMemory(rowPtr)
                            + vectorMemory(colIdx)
                            + vectorMemory(word) 
                            + sizeof(MatrixFormat) + sizeof(rows) + sizeof(cols) + sizeof(nnz);
            return static_cast<double>(memBytes) / (1024.0 * 1024.0);
        }
};

class MatrixBwcEll : public MatrixBase {
    public:
        uint32_t numWord, minNnzWord, maxNnzWord;
        double avgNnzWord;
        uint32_t k;
        std::vector<uint32_t> colIdx;
        std::vector<uint32_t> word;

        MatrixBwcEll();
        void print(const std::string& fileName) const override;
        uint32_t getNumWord() const override { return numWord; }
        uint32_t getMinNnzWord() const override { return minNnzWord; }
        uint32_t getMaxNnzWord() const override { return maxNnzWord; }
        double   getAvgNnzWord() const override { return avgNnzWord; }
        double memoryMB() const override {
            size_t memBytes = sizeof(k) + sizeof(numWord) + sizeof(minNnzWord) + sizeof(maxNnzWord) + sizeof(avgNnzWord)
                            + vectorMemory(colIdx)
                            + vectorMemory(word) 
                            + sizeof(MatrixFormat) + sizeof(rows) + sizeof(cols) + sizeof(nnz);
            return static_cast<double>(memBytes) / (1024.0 * 1024.0);
        }
};


class MatrixBwcHll : public MatrixBase {
    public:
        uint32_t numWord, minNnzWord, maxNnzWord;
        double avgNnzWord;
        uint32_t hackSize;
        std::vector<uint32_t> hack;
        std::vector<uint32_t> colIdx;
        std::vector<uint32_t> word;

        MatrixBwcHll();
        void print(const std::string& fileName) const override;
        
        uint32_t getNumWord() const override { return numWord; }
        uint32_t getMinNnzWord() const override { return minNnzWord; }
        uint32_t getMaxNnzWord() const override { return maxNnzWord; }
        double   getAvgNnzWord() const override { return avgNnzWord; }
        double memoryMB() const override {
            size_t memBytes = sizeof(hackSize) + sizeof(numWord) + sizeof(minNnzWord) + sizeof(maxNnzWord) + sizeof(avgNnzWord)
                            + vectorMemory(hack)
                            + vectorMemory(colIdx)
                            + vectorMemory(word) 
                            + sizeof(MatrixFormat) + sizeof(rows) + sizeof(cols) + sizeof(nnz);
            return static_cast<double>(memBytes) / (1024.0 * 1024.0);
        }
};


// ===== Matrix Wrapper Class =====

class Matrix {
    friend void spmv(const Matrix *A, const std::vector<uint32_t> *x, std::vector<uint32_t> *y, ExecutionMode executionMode);

private:
    MatrixBase *matrixImplementation;

public:
    MatrixFormat format;

    Matrix(const std::string& filename, MatrixFormat format);

    // void load(const std::string& filename, MatrixFormat format = MatrixFormat::COO);
    void print() const;
    MatrixFormat getFormat() const;
    uint32_t getRows() const;
    uint32_t getCols() const;
    uint32_t getNonZeros() const;

    // BWC word stats
    uint32_t getNumWord() const { return matrixImplementation->getNumWord(); }
    uint32_t getMinNnzWord() const { return matrixImplementation->getMinNnzWord(); }
    uint32_t getMaxNnzWord() const { return matrixImplementation->getMaxNnzWord(); }
    double   getAvgNnzWord() const { return matrixImplementation->getAvgNnzWord(); }
    double   getMemoryOccupationMB() const { return matrixImplementation->memoryMB(); }
};



#endif  
