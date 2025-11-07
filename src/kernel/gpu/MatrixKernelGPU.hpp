#ifndef MATRIX_KERNEL_GPU_HPP
#define MATRIX_KERNEL_GPU_HPP

#include "../MatrixKernel.hpp"


void spmvGpu(const MatrixBase& matrixBase, const std::vector<uint32_t> *x, std::vector<uint32_t> *y);

#endif