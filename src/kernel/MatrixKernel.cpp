#include "MatrixKernel.hpp"

 /**
 * @brief This function implements the wrapper of the sequential operation of
 * matrix-vector multiplication.
 *
 * @param A THe matrix to be multiplied
 * @param x The vector to multiply the matrix with
 * @param y The output vector to populate with the result of the multiplication
 * @param executionMode Whetere to execute on CPU or GPU
 */
void spmv(const Matrix *A, const std::vector<uint32_t> *x, std::vector<uint32_t> *y, ExecutionMode executionMode) {
    uint32_t colIdxWord, rowIdxWord;

    if (!A || !x || !y) {
        throw std::runtime_error("Null pointer passed to spmv.");
    }
    if (!A->matrixImplementation) {
        throw std::runtime_error("Matrix implementation is null.");
    }
    
    const MatrixBase& matrixBase = *(A->matrixImplementation);

    rowIdxWord = (matrixBase.rows + 31) / 32;
    colIdxWord = (matrixBase.cols + 31) / 32;

    if (x->size() != colIdxWord || y->size() != rowIdxWord) {
        throw std::runtime_error("Invalid vector size for SpMV!");
    }

    if(executionMode == ExecutionMode::CPU){
        spmvCpu(matrixBase, x, y);
    }else{
        spmvGpu(matrixBase, x ,y);
    }
}