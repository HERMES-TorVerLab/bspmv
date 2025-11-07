#include "MatrixKernelGPU.hpp"
#include "MatrixKernelGPU.cuh"

__global__ void dummyKernel() {};


 /**
 * @brief This function implements the wrapper of the sequential operation of
 * matrix-vector multiplication.
 *
 * @param A THe matrix to be multiplied
 * @param x The vector to multiply the matrix with
 * @param y The output vector to populate with the result of the multiplication
 */
void spmvGpu(const MatrixBase& matrixBase, const std::vector<uint32_t> *x, std::vector<uint32_t> *y) {
    switch (matrixBase.getFormat()) {
        case MatrixFormat::COO:
            spmvGpuCOO(static_cast<const MatrixCOO*>(&matrixBase), *x, *y);
            break;
        case MatrixFormat::CSR:
            spmvGpuCSR(static_cast<const MatrixCSR*>(&matrixBase), *x, *y);
            break;
        case MatrixFormat::ELL:
            spmvGpuELL(static_cast<const MatrixELL*>(&matrixBase), *x, *y);
            break;
        case MatrixFormat::HLL:
            spmvGpuHLL(static_cast<const MatrixHLL*>(&matrixBase), *x, *y);
            break;        
        case MatrixFormat::BWC_COO:
            spmvGpuBwcCoo(static_cast<const MatrixBwcCoo*>(&matrixBase), *x, *y);
            break;
        case MatrixFormat::BWC_CSR:
            spmvGpuBwcCsr(static_cast<const MatrixBwcCsr*>(&matrixBase), *x, *y);
            break;
        case MatrixFormat::BWC_ELL:
            spmvGpuBwcEll(static_cast<const MatrixBwcEll*>(&matrixBase), *x, *y);
            break;
        case MatrixFormat::BWC_HLL:
            spmvGpuBwcHll(static_cast<const MatrixBwcHll*>(&matrixBase), *x, *y);
            break;        
        default:
            throw std::runtime_error("Unsupported matrix type in spmv");
    }
}