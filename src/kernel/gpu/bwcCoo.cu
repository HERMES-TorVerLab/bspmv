/**
 * @file bwcCoo.cu
 * @author Stack1
 * @brief
 * @version 0.1
 * @date 17-02-2025
 *
 * Copyright (c) Simone Staccone, Tor Vergata University, Rome
 *
 */
#include "MatrixKernelGPU.cuh"                                                                                   
    

__global__ void spmvGpuBwcCooKernel(
    uint32_t * __restrict__ words, 
    uint32_t * __restrict__ rowIdx, 
    uint32_t * __restrict__ colIdx, 
    const uint32_t * __restrict__ x, 
    uint32_t *y, 
    int numWord)
{
    uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    uint32_t row, tmp;

    if (tid < numWord)
    {
        row = rowIdx[tid] - 1;

        /* Actual computation */
        tmp = words[tid] & __ldg(&x[colIdx[tid] - 1]);

        /* Parity check */
        tmp ^= tmp >> 16;
        tmp ^= tmp >> 8;
        tmp ^= tmp >> 4;
        tmp &= 0x0F; // Delete higer bits

        /* Create bitmask to identify the row in the word */
        tmp = (((0x6996 >> tmp) & 1U) << (row % 32));

        /* Atomically update the word relative to the correct row */
        atomicXor(&y[row / 32], tmp);
    }
}


void spmvGpuBwcCoo(const MatrixBwcCoo* A, const std::vector<uint32_t>& x, std::vector<uint32_t>& y)
{
    /* GPU variables */
    uint32_t * __restrict__ deviceA;
    uint32_t * __restrict__ deviceRowIdx;
    uint32_t * __restrict__ deviceColIdx;
    uint32_t * __restrict__ deviceX; 
    uint32_t *deviceY;

    /* TImers variables */
    cudaEvent_t start;
    cudaEvent_t stop;
    float elapsedTime, avgElapsedtime;

    /* Cuda kernel parametes */
    uint32_t nThreads = A->numWord;
    uint32_t blocks = (nThreads + THREAD_PER_BLOCK - 1) / THREAD_PER_BLOCK;

    /* Word index mapping variables */
    size_t numWordsX = (A->cols + 31) / 32;
    size_t numWordsY = (A->rows + 31) / 32;

    /* Initialize timers */
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);
    /* Allocate memory on device */
    CHECK_CUDA(cudaMalloc((void **)&deviceA, sizeof(uint32_t) * A->numWord));
    CHECK_CUDA(cudaMalloc((void **)&deviceRowIdx, sizeof(uint32_t) * A->numWord));
    CHECK_CUDA(cudaMalloc((void **)&deviceColIdx, sizeof(uint32_t) * A->numWord));
    CHECK_CUDA(cudaMalloc((void **)&deviceX, sizeof(uint32_t) * numWordsX));
    CHECK_CUDA(cudaMalloc((void **)&deviceY, sizeof(uint32_t) * numWordsY));

    /* Data transfer between host and device */
    CHECK_CUDA(cudaMemcpy(deviceA, A->word.data(), sizeof(uint32_t) * A->numWord, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(deviceRowIdx, A->rowIdx.data(), sizeof(uint32_t) * A->numWord, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(deviceColIdx, A->colIdx.data(), sizeof(uint32_t) * A->numWord, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(deviceX, x.data(), sizeof(uint32_t) * numWordsX, cudaMemcpyHostToDevice));

    cudaEventRecord(stop, 0);

    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    fprintf(stdout, "[INFO]\tTime to transfer data to device: %f ms\n", elapsedTime);


    dummyKernel<<<blocks,THREAD_PER_BLOCK>>>();

    avgElapsedtime = 0.0;

    for (int i = 0; i < LAUNCHES; i++)
    {
        cudaMemset(deviceY,0,sizeof(uint32_t) * numWordsY);

        cudaEventRecord(start, 0);
        spmvGpuBwcCooKernel<<<blocks, THREAD_PER_BLOCK>>>(deviceA, deviceRowIdx, deviceColIdx, deviceX, deviceY, A->numWord);
        cudaEventRecord(stop, 0);

        CHECK_CUDA(cudaPeekAtLastError());
        CHECK_CUDA(cudaDeviceSynchronize());

        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsedTime, start, stop);
        avgElapsedtime += elapsedTime;
    }

    avgElapsedtime = avgElapsedtime / LAUNCHES;

    std::cout << "[STAT]\tComputation time: " << avgElapsedtime <<"\n";


    cudaEventRecord(start, 0);
    CHECK_CUDA(cudaMemcpy(y.data(), deviceY, sizeof(uint32_t) * numWordsY, cudaMemcpyDeviceToHost));
    cudaEventRecord(stop, 0);

    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    fprintf(stdout, "[INFO]\tTime to transfer data from device to host: %f ms\n", elapsedTime);

    /* Destroy timers */
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    /* Deallocate memory */
    cudaFree(deviceA);
    cudaFree(deviceRowIdx);
    cudaFree(deviceColIdx);
    cudaFree(deviceX);
    cudaFree(deviceY);
}
