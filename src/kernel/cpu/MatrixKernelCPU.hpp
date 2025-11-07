#ifndef MATRIX_KERNEL_CPU_HPP
#define MATRIX_KERNEL_CPU_HPP

#include "../MatrixKernel.hpp"


void spmvCpu(const MatrixBase& matrixBase, const std::vector<uint32_t> *x, std::vector<uint32_t> *y);

#endif