/**
 * @file ell.cu
 * @author Stack1
 * @brief
 * @version 1.0
 * @date 17-03-2025
 *
 *
 */
#include "MatrixKernelGPU.cuh"



/* Column major */
__global__ void spmvEllKernel(
    uint32_t rows, 
    uint32_t k, 
    const uint32_t * __restrict__ colIdx, 
    const uint32_t * __restrict__ x, 
    uint32_t *y)
{
    uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    uint32_t rowWord = tid >> 5; // tid / 32
    uint32_t rowBit  = tid & 31; // tid % 32
    uint32_t tmp = 0;
    uint32_t wordX; // colIdx / 32
    uint32_t bitX; // colIdx % 32
    uint32_t col, xVal, mask;

    if (tid >= rows) return;


    for (uint32_t j = 0; j < k; j++)
    {
        col = colIdx[rows * j + tid]; 
        if (col == 0) break;          // skip row when encounter padding

        col -= 1; // convert to 0-based
        wordX = col >> 5; // col / 32
        bitX  = col & 31; // col % 32

        xVal = (__ldg(&x[wordX]) >> bitX) & 1U;
        tmp ^= xVal;
    }

    // Write bit-packed result
    mask = 1U << rowBit;
    if(tmp){
        atomicXor(&y[rowWord], mask);
    }
}


void spmvGpuELL(const MatrixELL* A, const std::vector<uint32_t>& x, std::vector<uint32_t>& y)
{
    /* Cuda kernel parametes */
    int nThreads, blocks;

    /* GPU variables */
    uint32_t * __restrict__ deviceColIdx;
    uint32_t * __restrict__ deviceX;
    uint32_t *deviceY;

    /* TImers variables */
    cudaEvent_t start;
    cudaEvent_t stop;
    float elapsed, avgElapsedTime;

    size_t numWordsX = (A->cols + 31) / 32; 
    size_t numWordsY = (A->rows + 31) / 32; 

    /* Initialize timers */
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);
    /* Allocate memory on device */
    CHECK_CUDA(cudaMalloc((void **)&deviceColIdx, sizeof(uint32_t) * A->k * A->rows));
    CHECK_CUDA(cudaMalloc((void **)&deviceX, sizeof(uint32_t) * numWordsX));
    CHECK_CUDA(cudaMalloc((void **)&deviceY, sizeof(uint32_t) * numWordsY));

    /* Data transfer from host to device */
    CHECK_CUDA(cudaMemcpy(deviceColIdx, A->colIdx.data(), sizeof(uint32_t) * A->k * A->rows, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(deviceX, x.data(), sizeof(uint32_t) * numWordsX, cudaMemcpyHostToDevice));

    cudaEventRecord(stop, 0);

    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed, start, stop);
    fprintf(stdout, "[INFO]\tTime to transfer data to device: %f ms\n", elapsed);

    nThreads = A->rows;
    blocks = (nThreads + THREAD_PER_BLOCK - 1)/THREAD_PER_BLOCK;


    dummyKernel<<<blocks, THREAD_PER_BLOCK>>>();

    /* Computational kernel excecution */
    avgElapsedTime = 0.0;

    for (int i = 0; i < LAUNCHES; i++)
    {
        CHECK_CUDA(cudaMemset(deviceY,0,numWordsY * sizeof(uint32_t)));
        
        cudaEventRecord(start, 0);
        spmvEllKernel<<<blocks, THREAD_PER_BLOCK>>>(A->rows, A->k, deviceColIdx, deviceX, deviceY);
        cudaEventRecord(stop, 0);

        CHECK_CUDA(cudaPeekAtLastError());
        CHECK_CUDA(cudaDeviceSynchronize());

        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsed, start, stop);
        avgElapsedTime += elapsed;
    }

    avgElapsedTime = avgElapsedTime / LAUNCHES;

    std::cout << "[STAT]\tComputation time: " << avgElapsedTime <<"\n";

    /* Data transfer from device to host */
    cudaEventRecord(start, 0);

    CHECK_CUDA(cudaMemcpy(y.data(), deviceY, numWordsY * sizeof(uint32_t), cudaMemcpyDeviceToHost));

    cudaEventRecord(stop, 0);

    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed, start, stop);
    fprintf(stdout, "[INFO]\tTime to transfer data to host: %f ms\n", elapsed);

    cudaFree(deviceColIdx);
    cudaFree(deviceX);
    cudaFree(deviceY);

}