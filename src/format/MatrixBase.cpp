/**
 * @file MatrixBase.cpp
 * @author Stack1
 * @brief File needed to provide the abstract class that later implements
 * all derived types.
 * @version 1.0
 * @date 25-08-2025
 * 
 * 
 */
#include "Matrix.hpp"


MatrixBase::~MatrixBase(){
};

MatrixFormat MatrixBase::getFormat() const{
    return this->format;
}

uint32_t MatrixBase::getRows() const {
    return this->rows;
}

uint32_t MatrixBase::getCols() const {
    return this->cols;
}

uint32_t MatrixBase::getNonZeros() const {
    return this->nnz;
}