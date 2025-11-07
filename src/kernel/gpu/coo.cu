/**
 * @file coo.cu
 * @author Stack1
 * @brief
 * @version 1.0
 * @date 25-02-2025
 *
 *
 */
#include "MatrixKernelGPU.cuh"


/**
 * @brief Naive kernel for COO SpMV having atomic updates
 * guarantieed by the hardware. (y = Ax)
 *
 * @param rows Rows indexes array
 * @param cols Column indexes array
 * @param x Vector to multiply with the matrix
 * @param y Return vector
 * @param size The number of non zero elements of the matrix
 */
__global__ void spmvCooKernel(
    const uint32_t *__restrict__ rows,
    const uint32_t *__restrict__ cols,
    const uint32_t *__restrict__ x,
    uint32_t *y,
    uint32_t nnz)
{
    uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    uint32_t row, col, xVal;
    int wordX, bitX, wordY, bitY;

    if (tid < nnz)
    {
        row = rows[tid] - 1;
        col = cols[tid] - 1;

        wordX = col >> 5;        // col / 32
        bitX  = col & 31;        // col % 32

        xVal  = (__ldg(&x[wordX]) >> bitX) & 1U;   // extract the single bit

        if (xVal) {
            wordY = row >> 5;                  // row / 32
            bitY  = row & 31;                  // row % 32
            uint32_t mask = (1U << bitY);
            atomicXor(&y[wordY], mask);
        }
    }
}


/**
 * @brief CUDA wrapper used to call the cuda kernel.+
 *
 * @param A Matrix A in COO format
 * @param x Vector to be mutliplied by
 * @param y Resutly vector
 */
void spmvGpuCOO(const MatrixCOO* A, const std::vector<uint32_t>& x, std::vector<uint32_t>& y)
{
    /* GPU variables */
    uint32_t *rowIdx;
    uint32_t *colIdx;
    uint32_t *deviceX;
    uint32_t *deviceY;

    /* TImers variables */
    cudaEvent_t start;
    cudaEvent_t stop;
    float elapsedTime, avgelapsedTime;

    /* Kernel variables */
    uint32_t nThreads = A->nnz;
    uint32_t blocks = (nThreads + THREAD_PER_BLOCK - 1) / THREAD_PER_BLOCK;

    /* Word index mapping variables */
    size_t numWordsX = (A->cols + 31) / 32;
    size_t numWordsY = (A->rows + 31) / 32;

    /* Initialize timers */
    cudaEventCreate(&start);
    cudaEventCreate(&stop);


    cudaEventRecord(start, 0);
    /* Allocate memory on device */
    CHECK_CUDA(cudaMalloc((void **)&rowIdx, sizeof(uint32_t) * A->nnz));
    CHECK_CUDA(cudaMalloc((void **)&colIdx, sizeof(uint32_t) * A->nnz));
    CHECK_CUDA(cudaMalloc((void **)&deviceX, sizeof(uint32_t) * numWordsX));
    CHECK_CUDA(cudaMalloc((void **)&deviceY, sizeof(uint32_t) * numWordsY));

    /* Data transfer from host to device */
    CHECK_CUDA(cudaMemcpy(rowIdx, A->rowIdx.data(), sizeof(uint32_t) * A->nnz, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(colIdx, A->colIdx.data(), sizeof(uint32_t) * A->nnz, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(deviceX, x.data(), sizeof(uint32_t) * numWordsX, cudaMemcpyHostToDevice));

    cudaEventRecord(stop, 0);

    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    fprintf(stdout, "[INFO]\tTime to transfer data to device: %f ms\n", elapsedTime);

    /* Computational kernel excecution */
    avgelapsedTime = 0.0;

    /* First kernel launch to avoid collecting drivers loading in metrics */
    dummyKernel<<<blocks, THREAD_PER_BLOCK>>>();

    for (int i = 0; i < LAUNCHES; i++)
    {
        CHECK_CUDA(cudaMemset(deviceY, 0U , sizeof(uint32_t) * numWordsY));
        
        cudaEventRecord(start, 0);
        spmvCooKernel<<<blocks, THREAD_PER_BLOCK>>>(rowIdx, colIdx, deviceX, deviceY, A->nnz);
        cudaEventRecord(stop, 0);

        CHECK_CUDA(cudaPeekAtLastError());
        CHECK_CUDA(cudaDeviceSynchronize());

        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsedTime, start, stop);
        avgelapsedTime += elapsedTime;
    }

    avgelapsedTime = avgelapsedTime / LAUNCHES;

    std::cout << "[STAT]\tComputation time: " << avgelapsedTime <<"\n";

    /* Data transfer from device to host */
    cudaEventRecord(start, 0);
    CHECK_CUDA(cudaMemcpy(y.data(), deviceY, sizeof(uint32_t) * numWordsY, cudaMemcpyDeviceToHost));
    cudaEventRecord(stop, 0);

    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    fprintf(stdout, "[INFO]\tTime to transfer data to host: %f ms\n", elapsedTime);


    cudaFree(rowIdx);
    cudaFree(colIdx);
    cudaFree(deviceX);
    cudaFree(deviceY);

}