/**
 * @file hll.cu
 * @author Stack1
 * @brief 
 * @version 1.0
 * @date 16-09-2025
 * 
 * 
 */

#include "MatrixKernelGPU.cuh"

__global__ void spmvHllKernel(
    const uint32_t* __restrict__ colIdx,
    const uint32_t* __restrict__ hack,
    const uint32_t* __restrict__ x,
    uint32_t* y,
    uint32_t numRows,
    uint32_t hackSize)
{
    uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= numRows) return;

    uint32_t rowWord = tid >> 5; // tid / 32
    uint32_t rowBit  = tid & 31; // tid % 32

    // Find which hack this row belongs to
    uint32_t currHackIdx = tid / hackSize;
    uint32_t localRow = tid % hackSize;

    // Use hack to locate the start of this hack
    uint32_t offsetStart = hack[currHackIdx];
    uint32_t offsetEnd   = hack[currHackIdx + 1];
    uint32_t localRows   = min(hackSize, numRows - currHackIdx * hackSize);
    uint32_t hackWidth   = (offsetEnd - offsetStart) / localRows;

    uint32_t tmp = 0;

    for (uint32_t j = 0; j < hackWidth; ++j) {
        uint32_t idx = offsetStart + j * localRows + localRow;
        uint32_t col = colIdx[idx];
        if (col == 0) continue;  // skip padding
        col -= 1;

        uint32_t wordX = col >> 5;
        uint32_t bitX  = col & 31;
        uint32_t xVal  = (__ldg(&x[wordX]) >> bitX) & 1U;
        tmp ^= xVal;
    }

    if (tmp) {
        atomicXor(&y[rowWord], 1U << rowBit);
    }
}




 void spmvGpuHLL(const MatrixHLL* A, const std::vector<uint32_t>& x, std::vector<uint32_t>& y)
{
    int nThreads, blocks;

    uint32_t* deviceColIdx;
    uint32_t* deviceHack;
    uint32_t* deviceX;
    uint32_t* deviceY;

    cudaEvent_t start, stop;
    float elapsed, avgElapsedTime;

    size_t numWordsX = (A->cols + 31) / 32;
    size_t numWordsY = (A->rows + 31) / 32;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);

    CHECK_CUDA(cudaMalloc((void**)&deviceColIdx, sizeof(uint32_t) * A->colIdx.size()));
    CHECK_CUDA(cudaMalloc((void**)&deviceHack, sizeof(uint32_t) * A->hack.size()));
    CHECK_CUDA(cudaMalloc((void**)&deviceX, sizeof(uint32_t) * numWordsX));
    CHECK_CUDA(cudaMalloc((void**)&deviceY, sizeof(uint32_t) * numWordsY));

    CHECK_CUDA(cudaMemcpy(deviceColIdx, A->colIdx.data(), sizeof(uint32_t) * A->colIdx.size(), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(deviceHack, A->hack.data(), sizeof(uint32_t) * A->hack.size(), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(deviceX, x.data(), sizeof(uint32_t) * numWordsX, cudaMemcpyHostToDevice));

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed, start, stop);
    fprintf(stdout, "[INFO] Time to transfer data to device: %f ms\n", elapsed);

    nThreads = A->rows;
    blocks = (nThreads + THREAD_PER_BLOCK - 1) / THREAD_PER_BLOCK;

    dummyKernel<<<blocks, THREAD_PER_BLOCK>>>();  // optional warm-up

    avgElapsedTime = 0.0;
    for (int i = 0; i < LAUNCHES; i++) {
        CHECK_CUDA(cudaMemset(deviceY, 0, sizeof(uint32_t) * numWordsY));

        cudaEventRecord(start, 0);
        spmvHllKernel<<<blocks, THREAD_PER_BLOCK>>>(deviceColIdx, deviceHack, deviceX, deviceY, A->rows, A->hackSize);
        cudaEventRecord(stop, 0);

        CHECK_CUDA(cudaPeekAtLastError());
        CHECK_CUDA(cudaDeviceSynchronize());

        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsed, start, stop);
        avgElapsedTime += elapsed;
    }

    avgElapsedTime /= LAUNCHES;
    std::cout << "[STAT] Computation time: " << avgElapsedTime << "\n";

    cudaEventRecord(start, 0);
    CHECK_CUDA(cudaMemcpy(y.data(), deviceY, sizeof(uint32_t) * numWordsY, cudaMemcpyDeviceToHost));
    cudaEventRecord(stop, 0);

    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed, start, stop);
    fprintf(stdout, "[INFO] Time to transfer data to host: %f ms\n", elapsed);

    // Cleanup
    cudaFree(deviceColIdx);
    cudaFree(deviceHack);
    cudaFree(deviceX);
    cudaFree(deviceY);
}
