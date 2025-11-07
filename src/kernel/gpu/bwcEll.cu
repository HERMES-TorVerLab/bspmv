/**
 * @file bwcEll.cu
 * @author Stack1
 * @brief 
 * @version 1.0
 * @date 15-09-2025
 * 
 * 
 */
 #include "MatrixKernelGPU.cuh"


/* Column major */
__global__ void spmvBwcEllKernel(
    uint32_t * __restrict__ word, 
    uint32_t * __restrict__ colIdx, 
    const uint32_t * __restrict__ x, 
    uint32_t *y,
    uint32_t numRows, 
    uint32_t k)
{
    uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    uint32_t tmp, globalTmp, col;

    if (tid < numRows)
    {

        globalTmp = 0U;
        tmp = 0U;

        for (uint32_t j = 0; j < k; ++j)
        {
            col = colIdx[numRows * j + tid]; 
            if (col == 0) break;          // skip row when encounter padding

            col -=1;
            tmp = word[numRows * j + tid] & __ldg(&x[col]);

            /* Check for parity */
            tmp ^= tmp >> 16;
            tmp ^= tmp >> 8;
            tmp ^= tmp >> 4;
            tmp &= 0x0F;   

            /* Set the right word index */
            tmp = (((0x6996 >> tmp) & 1) << (tid % 32));   

            globalTmp ^= tmp;
            
        }
        // atomicXor(&y[tid / 32],globalTmp);
        
        /* Use these to avoid using atomic operation and prefer warp level atomics */
        globalTmp ^= __shfl_down_sync(0xffffffff, globalTmp, 16);
        globalTmp ^= __shfl_down_sync(0xffffffff, globalTmp, 8);
        globalTmp ^= __shfl_down_sync(0xffffffff, globalTmp, 4);
        globalTmp ^= __shfl_down_sync(0xffffffff, globalTmp, 2);
        globalTmp ^= __shfl_down_sync(0xffffffff, globalTmp, 1);
        if (tid % 32 == 0)
            y[tid / 32] = globalTmp;
    }
}



void spmvGpuBwcEll(const MatrixBwcEll* A, const std::vector<uint32_t>& x, std::vector<uint32_t>& y)
{
    /* Cuda kernel parametes */
    int nThreads, blocks;

    /* GPU variables */
    uint32_t * __restrict__ deviceA;
    uint32_t * __restrict__ deviceColIdx;
    uint32_t * __restrict__ deviceX;
    uint32_t *deviceY;

    /* TImers variables */
    cudaEvent_t start;
    cudaEvent_t stop;
    float elapsedTime, avgElapsedTime;

    size_t numWordsX = (A->cols + 31) / 32; 
    size_t numWordsY = (A->rows + 31) / 32; 

    /* Initialize timers */
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);

    /* Allocate memory on device */
    CHECK_CUDA(cudaMalloc((void **)&deviceA, sizeof(uint32_t) * A->numWord));
    CHECK_CUDA(cudaMalloc((void **)&deviceColIdx, sizeof(uint32_t) * A->numWord));
    CHECK_CUDA(cudaMalloc((void **)&deviceX, sizeof(uint32_t) * numWordsX));
    CHECK_CUDA(cudaMalloc((void **)&deviceY, sizeof(uint32_t) * numWordsY));

    /* Data transfer from host to device */
    CHECK_CUDA(cudaMemcpy(deviceA, A->word.data(), sizeof(uint32_t) * A->numWord, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(deviceColIdx, A->colIdx.data(), sizeof(uint32_t) * A->k * A->rows, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(deviceX, x.data(), sizeof(uint32_t) * numWordsX, cudaMemcpyHostToDevice));

    cudaEventRecord(stop, 0);

    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    fprintf(stdout, "[INFO]\tTime to transfer data to device: %f ms\n", elapsedTime);

    nThreads = A->rows;
    blocks = (nThreads + THREAD_PER_BLOCK - 1)/THREAD_PER_BLOCK;


    dummyKernel<<<blocks, THREAD_PER_BLOCK>>>();

    /* Computational kernel excecution */
    avgElapsedTime = 0.0;

    for (int i = 0; i < LAUNCHES; i++)
    {
        CHECK_CUDA(cudaMemset(deviceY,0,numWordsY * sizeof(uint32_t)));
        
        cudaEventRecord(start, 0);
        spmvBwcEllKernel<<<blocks, THREAD_PER_BLOCK>>>(deviceA, deviceColIdx, deviceX, deviceY, A->rows, A->k);
        cudaEventRecord(stop, 0);

        CHECK_CUDA(cudaPeekAtLastError());
        CHECK_CUDA(cudaDeviceSynchronize());

        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsedTime, start, stop);
        avgElapsedTime += elapsedTime;
    }

    avgElapsedTime = avgElapsedTime / LAUNCHES;

    std::cout << "[STAT]\tComputation time: " << avgElapsedTime <<"\n";


    /* Data transfer from device to host */
    cudaEventRecord(start, 0);

    CHECK_CUDA(cudaMemcpy(y.data(), deviceY, numWordsY * sizeof(uint32_t), cudaMemcpyDeviceToHost));

    cudaEventRecord(stop, 0);

    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    fprintf(stdout, "[INFO]\tTime to transfer data to host: %f ms\n", elapsedTime);

    CHECK_CUDA(cudaFree(deviceA));
    CHECK_CUDA(cudaFree(deviceColIdx));
    CHECK_CUDA(cudaFree(deviceX));
    CHECK_CUDA(cudaFree(deviceY));

}