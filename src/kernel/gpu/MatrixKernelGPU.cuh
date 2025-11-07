#ifndef MATRIX_KERNEL_GPU_CUH
#define MATRIX_KERNEL_GPU_CUH

#include "../MatrixKernel.hpp"

#define THREAD_PER_BLOCK 512
#define WARP_SIZE 4
#define LAUNCHES 100

inline void checkCuda(cudaError_t error, const char* file, int line)
{
    if (error != cudaSuccess)
    {
        std::cout << "[ERROR] File: " << file
                  << ", Line: " << line
                  << ", Code: " << error
                  << ", Reason: " << cudaGetErrorString(error);

        // Append specific message
        switch (error)
        {
            case cudaErrorMemoryAllocation:
                std::cout << " -> Out of GPU memory.";
                break;
            case cudaErrorInvalidValue:
                std::cout << " -> Invalid value passed to a CUDA function.";
                break;
            case cudaErrorInvalidDevice:
                std::cout << " -> Invalid device ID (GPU may not exist).";
                break;
            case cudaErrorLaunchFailure:
                std::cout << " -> Kernel launch failed (possible illegal memory access).";
                break;
            case cudaErrorLaunchTimeout:
                std::cout << " -> Kernel execution timed out.";
                break;
            case cudaErrorNoDevice:
                std::cout << " -> No CUDA-capable device found.";
                break;
            case cudaErrorUnknown:
                std::cout << " -> Unknown CUDA error.";
                break;
            default:
                std::cout << " -> Unhandled CUDA error.";
        }

        // Reset color and terminate line
        std::cout << std::endl;

        // Reset CUDA error state
        cudaGetLastError();

        // Stop execution
        exit(1);
    }
}

// Helper macro to preserve file/line info like before
#define CHECK_CUDA(call) checkCuda((call), __FILE__, __LINE__)

__global__ void dummyKernel();

void spmvGpuCOO(const MatrixCOO* A, const std::vector<uint32_t>& x, std::vector<uint32_t>& y);
void spmvGpuCSR(const MatrixCSR* A, const std::vector<uint32_t>& x, std::vector<uint32_t>& y);
void spmvGpuELL(const MatrixELL* A, const std::vector<uint32_t>& x, std::vector<uint32_t>& y);
void spmvGpuHLL(const MatrixHLL* A, const std::vector<uint32_t>& x, std::vector<uint32_t>& y);

void spmvGpuBwcCoo(const MatrixBwcCoo* A, const std::vector<uint32_t>& x, std::vector<uint32_t>& y);
void spmvGpuBwcCsr(const MatrixBwcCsr* A, const std::vector<uint32_t>& x, std::vector<uint32_t>& y);
void spmvGpuBwcEll(const MatrixBwcEll* A, const std::vector<uint32_t>& x, std::vector<uint32_t>& y);
void spmvGpuBwcHll(const MatrixBwcHll* A, const std::vector<uint32_t>& x, std::vector<uint32_t>& y);

#endif