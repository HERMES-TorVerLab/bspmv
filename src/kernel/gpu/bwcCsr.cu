/**
 * @file bwcCsr.cu
 * @author Stack1
 * @brief 
 * @version 1.0
 * @date 14-09-2025
 * 
 * 
 */

#include "MatrixKernelGPU.cuh"

__global__ void spmvGpuCsrVectorKernel(uint32_t *A, uint32_t *rowPtr, uint32_t *colIdx, uint32_t *x, uint32_t *y, uint32_t rows)
{
    uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    uint32_t warpId = tid / WARP_SIZE;
    uint32_t lane = tid % WARP_SIZE; // Thread index in the warp
    uint32_t tmp, globalTmp;
    uint32_t rowStart, rowEnd;
    __shared__ volatile uint32_t sharedVals[THREAD_PER_BLOCK];

    if (warpId < rows)
    {
        tmp = 0;
        globalTmp = 0U;
        rowStart = rowPtr[warpId];
        rowEnd = rowPtr[warpId + 1];

        for (uint32_t j = rowStart + lane; j < rowEnd; j += WARP_SIZE)
        {
            tmp = A[j] & x[colIdx[j] - 1];

            /* Check for parity */
            tmp ^= tmp >> 16;
            tmp ^= tmp >> 8;
            tmp ^= tmp >> 4;
            tmp &= 0x0F;

            /* Set the right word index */
            tmp = (((0x6996 >> tmp) & 1U) << (warpId % 32));

            globalTmp ^= tmp;
        }
        sharedVals[threadIdx.x] = globalTmp;
        __syncthreads();


        /* Warp reduction */
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
            atomicXor(&y[warpId / 32], sharedVals[threadIdx.x]);
        }
    }
}

__global__ void spmvGpuCsrScalarKernel(uint32_t *A, uint32_t *rowPtr, uint32_t *colidx, uint32_t *x, uint32_t *y, uint32_t rows)
{
    uint32_t tmp, globalTmp, rowStart, rowEnd;
    uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;

    if (tid < rows)
    {
        globalTmp = 0U;
        tmp = 0U;

        rowStart = rowPtr[tid];
        rowEnd = rowPtr[tid + 1];

        for (uint32_t j = rowStart; j < rowEnd; j++)
        {
            tmp = A[j] & x[colidx[j] - 1];

            /* Check for parity */
            tmp ^= tmp >> 16;
            tmp ^= tmp >> 8;
            tmp ^= tmp >> 4;
            tmp &= 0x0F;

            /* Set the right word index */
            tmp = (((0x6996 >> tmp) & 1) << (tid % 32));

            globalTmp ^= tmp;
        }
        /* Warp reduction */
        // atomicXor(&y[tid / 32], global_tmp);
        globalTmp ^= __shfl_down_sync(0xffffffff, globalTmp, 16);
        globalTmp ^= __shfl_down_sync(0xffffffff, globalTmp, 8);
        globalTmp ^= __shfl_down_sync(0xffffffff, globalTmp, 4);
        globalTmp ^= __shfl_down_sync(0xffffffff, globalTmp, 2);
        globalTmp ^= __shfl_down_sync(0xffffffff, globalTmp, 1);

        if (tid % 32 == 0)
            y[tid / 32] = globalTmp;
    }
}


void spmvGpuBwcCsr(const MatrixBwcCsr* A, const std::vector<uint32_t>& x, std::vector<uint32_t>& y)
{
    /* GPU variables */
    uint32_t * __restrict__ deviceA;
    uint32_t * __restrict__ deviceRowPtr;
    uint32_t * __restrict__ deviceColIdx;
    uint32_t * __restrict__ deviceX; 
    uint32_t *deviceYScalar, *deviceYVector;

    /* TImers variables */
    cudaEvent_t start;
    cudaEvent_t stop;
    float elapsedTime, avgElapsedTime;

    /* Cuda kernel parametes */
    uint32_t nThreads = A->rows;
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
    CHECK_CUDA(cudaMalloc((void **)&deviceRowPtr, sizeof(uint32_t) * (A->rows + 1) ));
    CHECK_CUDA(cudaMalloc((void **)&deviceColIdx, sizeof(uint32_t) * A->numWord));
    CHECK_CUDA(cudaMalloc((void **)&deviceX, sizeof(uint32_t) * numWordsX));
    CHECK_CUDA(cudaMalloc((void**)&deviceYScalar, sizeof(uint32_t) * numWordsY));
    CHECK_CUDA(cudaMalloc((void**)&deviceYVector, sizeof(uint32_t) * numWordsY));

    /* Data transfer between host and device */
    CHECK_CUDA(cudaMemcpy(deviceA, A->word.data(), sizeof(uint32_t) * A->numWord, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(deviceRowPtr, A->rowPtr.data(), sizeof(uint32_t) * (A->rows + 1), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(deviceColIdx, A->colIdx.data(), sizeof(uint32_t) * A->numWord, cudaMemcpyHostToDevice));
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
        spmvGpuCsrScalarKernel<<<blocks, THREAD_PER_BLOCK>>>(deviceA, deviceRowPtr, deviceColIdx, deviceX, deviceYScalar, A->rows);
        cudaEventRecord(stop, 0);

        CHECK_CUDA(cudaPeekAtLastError());
        CHECK_CUDA(cudaDeviceSynchronize());

        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsedTime, start, stop);
        avgElapsedTime += elapsedTime;
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
        spmvGpuCsrVectorKernel<<<blocks, THREAD_PER_BLOCK>>>(deviceA, deviceRowPtr, deviceColIdx, deviceX, deviceYVector, A->rows);
        cudaEventRecord(stop, 0);

        CHECK_CUDA(cudaPeekAtLastError());
        CHECK_CUDA(cudaDeviceSynchronize());

        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsedTime, start, stop);
        avgElapsedTime += elapsedTime;
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
