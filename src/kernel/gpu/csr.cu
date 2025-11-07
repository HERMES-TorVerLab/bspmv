/**
 * @file csr_gpu.cu
 * @author Stack1
 * @brief
 * @version 1.0
 * @date 06-03-2025
 *
 *
 */
#include "MatrixKernelGPU.cuh"

__global__ void spmvCsrVectorKernel(
    const uint32_t *__restrict__ rowPtr, 
    const uint32_t *__restrict__ colIdx, 
    const uint32_t *__restrict__ x, 
    uint32_t *y,
    uint32_t numRows)
{
    int threadId = threadIdx.x + blockIdx.x * blockDim.x;
    int warpId = threadId / WARP_SIZE;
    uint32_t lane = threadId % WARP_SIZE;  // Thread index in warp

    __shared__ volatile uint32_t sharedVals[THREAD_PER_BLOCK];

    if (warpId < numRows)
    {
        uint32_t tmp = 0;
        uint32_t rowStart = rowPtr[warpId];
        uint32_t rowEnd   = rowPtr[warpId + 1];

        // Each thread in the warp processes a subset of nonzeros
        for (uint32_t j = rowStart + lane; j < rowEnd; j += WARP_SIZE)
        {
            uint32_t col = colIdx[j] - 1;          
            uint32_t wordX = col >> 5;              // col / 32
            uint32_t bitX  = col & 31;              // col % 32
            uint32_t xVal  = (__ldg(&x[wordX]) >> bitX) & 1U;
            tmp ^= xVal;
        }

        // Store partial result in shared memory
        sharedVals[threadIdx.x] = tmp;
        __syncthreads();

        // Warp reduction
        for (uint32_t offset = WARP_SIZE / 2; offset > 0; offset /= 2)
        {
            if (lane < offset)
            {
                sharedVals[threadIdx.x] ^= sharedVals[threadIdx.x + offset];
            }
        }
        __syncthreads();

        if (lane == 0)
        {
            uint32_t rowWord = warpId >> 5;        // y word index
            uint32_t rowBit  = warpId & 31;        // bit in word
            uint32_t mask    = sharedVals[threadIdx.x] ? (1U << rowBit) : 0U;
            atomicXor(&y[rowWord], mask);
        }
    }
}


__global__ void spmvCsrScalarKernel(
    const uint32_t *__restrict__ rowPtr, 
    const uint32_t *__restrict__ colIdx, 
    const uint32_t *__restrict__ x, 
    uint32_t *y,
    uint32_t numRows)
{
    uint32_t rowStart, rowEnd, rowWord, rowBit, mask, tmp, col, wordX, bitX, xVal;
    uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;

    if (tid < numRows)
    {
        rowWord = tid >> 5;   // word in y
        rowBit  = tid & 31;   // bit in word
        mask = 1U << rowBit;  // mask for this row
        tmp = 0;

        rowStart = rowPtr[tid];
        rowEnd   = rowPtr[tid + 1];

        for (uint32_t j = rowStart; j < rowEnd; j++)
        {
            col    = colIdx[j] - 1;       // convert 1-based -> 0-based
            wordX  = col >> 5;            // word in x
            bitX   = col & 31;            // bit in word
            xVal  = (__ldg(&x[wordX]) >> bitX) & 1U;
            tmp ^= xVal;
        }

        if (tmp)
        {
            atomicXor(&y[rowWord], mask);
        }
    }
}


void spmvGpuCSR(const MatrixCSR* A, const std::vector<uint32_t>& x, std::vector<uint32_t>& y)
{
    /* GPU variables */
    uint32_t *deviceRowPtr;
    uint32_t *deviceColIdx;
    uint32_t *deviceX; 
    uint32_t *deviceYScalar, *deviceYVector;
    
    /* Kernel variables */
    int nThreads, blocks;

    /* TImers variables */
    cudaEvent_t start;
    cudaEvent_t stop;
    float elapsed, avgElapsedTime;

    size_t numWordsX = (A->cols + 31) / 32;
    size_t numWordsY = (A->rows + 31) / 32;

    /* Initialize timers */
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    /* Allocate device memory */
    CHECK_CUDA(cudaMalloc((void**)&deviceRowPtr, sizeof(uint32_t) * (A->rows + 1)));
    CHECK_CUDA(cudaMalloc((void**)&deviceColIdx, sizeof(uint32_t) * A->nnz));
    CHECK_CUDA(cudaMalloc((void**)&deviceX, sizeof(uint32_t) * numWordsX));
    CHECK_CUDA(cudaMalloc((void**)&deviceYScalar, sizeof(uint32_t) * numWordsY));
    CHECK_CUDA(cudaMalloc((void**)&deviceYVector, sizeof(uint32_t) * numWordsY));

    /* Copy data to device */
    CHECK_CUDA(cudaMemcpy(deviceRowPtr, A->rowPtr.data(), sizeof(uint32_t) * (A->rows + 1), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(deviceColIdx, A->colIdx.data(), sizeof(uint32_t) * A->nnz, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(deviceX, x.data(), sizeof(uint32_t) * numWordsX, cudaMemcpyHostToDevice));

    /* --- SCALAR KERNEL --- */
    nThreads = A->rows;
    blocks = (nThreads + THREAD_PER_BLOCK - 1) / THREAD_PER_BLOCK;

    dummyKernel<<<blocks,THREAD_PER_BLOCK>>>();
    avgElapsedTime = 0.0;

    for (int i = 0; i < LAUNCHES; i++)
    {
        CHECK_CUDA(cudaMemset(deviceYScalar, 0, numWordsY * sizeof(uint32_t)));
        
        cudaEventRecord(start, 0);
        spmvCsrScalarKernel<<<blocks, THREAD_PER_BLOCK>>>(deviceRowPtr, deviceColIdx, deviceX, deviceYScalar, A->rows);
        cudaEventRecord(stop, 0);

        CHECK_CUDA(cudaPeekAtLastError());
        CHECK_CUDA(cudaDeviceSynchronize());

        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsed, start, stop);
        avgElapsedTime += elapsed;
    }

    avgElapsedTime = avgElapsedTime / LAUNCHES;

    std::cout << "[STAT]\tScalar computation time: " << avgElapsedTime << "\n";

    /* --- VECTOR KERNEL --- */
    nThreads = A->rows * WARP_SIZE;
    blocks = (nThreads + THREAD_PER_BLOCK - 1) / THREAD_PER_BLOCK;


    dummyKernel<<<blocks,THREAD_PER_BLOCK>>>();
    avgElapsedTime = 0.0;

    for (int i = 0; i < LAUNCHES; i++)
    {
        CHECK_CUDA(cudaMemset(deviceYVector, 0, numWordsY * sizeof(uint32_t)));
        
        cudaEventRecord(start, 0);
        spmvCsrVectorKernel<<<blocks, THREAD_PER_BLOCK>>>(deviceRowPtr, deviceColIdx, deviceX, deviceYVector, A->rows);
        cudaEventRecord(stop, 0);

        CHECK_CUDA(cudaPeekAtLastError());
        CHECK_CUDA(cudaDeviceSynchronize());

        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsed, start, stop);
        avgElapsedTime += elapsed;
    }

    avgElapsedTime = avgElapsedTime / LAUNCHES;

    std::cout << "[STAT]\tVector computation time: " << avgElapsedTime << "\n";


    /* Copy scalar and vector results back to host for comparison */
    std::vector<uint32_t> yScalar(numWordsY);
    std::vector<uint32_t> yVector(numWordsY);
    CHECK_CUDA(cudaMemcpy(yScalar.data(), deviceYScalar, numWordsY * sizeof(uint32_t), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(yVector.data(), deviceYVector, numWordsY * sizeof(uint32_t), cudaMemcpyDeviceToHost));

    /* Compare results */
    bool match = true;
    for (size_t i = 0; i < numWordsY; i++){
        if (yScalar[i] != yVector[i])
        {
            match = false;
            std::cout << i << "\n";
            break;
        }
    }

    if (!match)
        std::cerr << "[ERROR]\tScalar and vector results differ!\n";
    else
        std::cout << "[INFO]\tScalar and vector results match.\n";

    /* Copy final result (vector) to output y */
    y.resize(numWordsY);
    std::copy(yVector.begin(), yVector.end(), y.begin());

    /* Free device memory */
    CHECK_CUDA(cudaFree(deviceRowPtr));
    CHECK_CUDA(cudaFree(deviceColIdx));
    CHECK_CUDA(cudaFree(deviceX));
    CHECK_CUDA(cudaFree(deviceYScalar));
    CHECK_CUDA(cudaFree(deviceYVector));
}

