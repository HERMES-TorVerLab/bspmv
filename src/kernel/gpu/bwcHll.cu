/**
 * @file bwcHll.cu
 * @author Stack1
 * @brief 
 * @version 1.0
 * @date 17-09-2025
 * 
 * 
 */


  #include "MatrixKernelGPU.cuh"



__global__ void spmvBwcHllKernel(
    const uint32_t* __restrict__ word,    // word masks, column-major per hack
    const uint32_t* __restrict__ colIdx,  // 1-based word indices (0 = padding)
    const uint32_t* __restrict__ hack,    // hack offsets (size = numHacks+1)
    const uint32_t* __restrict__ x,       // input vector packed as 32-bit words
    uint32_t* y,                          // output packed 32-bit words
    uint32_t numRows,
    uint32_t hackSize)
{
    uint32_t tmp, globalTmp, idx, col, offsetStart, offsetEnd, hackWidth, currHackIdx, localRow;
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;


    if (tid >= numRows) return;

    currHackIdx = tid / hackSize;
    localRow    = tid % hackSize;

    offsetStart = hack[currHackIdx];
    offsetEnd   = hack[currHackIdx + 1];
    hackWidth   = (offsetEnd - offsetStart) / hackSize;

    globalTmp = 0U;
    for (uint32_t j = 0; j < hackWidth; ++j) {
        idx = offsetStart + j * hackSize + localRow; // column-major
        col = colIdx[idx];

        if (col == 0) break; // padding
        col -= 1;               // convert to 0-based word index

        tmp = word[idx] & __ldg(&x[col]);

        /* Check for parity */
        tmp ^= tmp >> 16;
        tmp ^= tmp >> 8;
        tmp ^= tmp >> 4;
        tmp &= 0xF;
        
        /* Set the right word index */
        tmp = (((0x6996 >> tmp) & 1) << (tid % 32));        
        globalTmp ^= tmp;
    }

    // warp-wide XOR reduction, lane 0 gets packed bits
    globalTmp ^= __shfl_down_sync(0xffffffff, globalTmp, 16);
    globalTmp ^= __shfl_down_sync(0xffffffff, globalTmp, 8);
    globalTmp ^= __shfl_down_sync(0xffffffff, globalTmp, 4);
    globalTmp ^= __shfl_down_sync(0xffffffff, globalTmp, 2);
    globalTmp ^= __shfl_down_sync(0xffffffff, globalTmp, 1);

    if ((tid & 31U) == 0) 
        y[tid >> 5] = globalTmp;
    
}

void spmvGpuBwcHll(const MatrixBwcHll* A, const std::vector<uint32_t>& x, std::vector<uint32_t>& y)
{
    /* Cuda kernel parameters */
    int nThreads, blocks;

    /* GPU variables */
    uint32_t * __restrict__ deviceWord; 
    uint32_t * __restrict__ deviceColIdx;
    uint32_t * __restrict__ deviceHack;
    uint32_t * __restrict__ deviceX;
    uint32_t *deviceY;

    /* Timer variables */
    cudaEvent_t start;
    cudaEvent_t stop;
    float elapsedTime, avgElapsedTime;

    const uint32_t rows = A->rows;
    const uint32_t cols = A->cols;

    size_t numWordsX = (cols + 31) / 32;
    size_t numWordsY = (rows + 31) / 32;

    /* Initialize timers */
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);

    /* Allocate memory on device */
    CHECK_CUDA(cudaMalloc((void **)&deviceWord, sizeof(uint32_t) * A->numWord));
    CHECK_CUDA(cudaMalloc((void **)&deviceColIdx, sizeof(uint32_t) * A->numWord));
    CHECK_CUDA(cudaMalloc((void **)&deviceHack, sizeof(uint32_t) * A->hack.size()));
    CHECK_CUDA(cudaMalloc((void **)&deviceX, sizeof(uint32_t) * numWordsX));
    CHECK_CUDA(cudaMalloc((void **)&deviceY, sizeof(uint32_t) * numWordsY));

    /* Data transfer from host to device */
    CHECK_CUDA(cudaMemcpy(deviceWord, A->word.data(), sizeof(uint32_t) * A->numWord, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(deviceColIdx, A->colIdx.data(), sizeof(uint32_t) * A->numWord, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(deviceHack, A->hack.data(), sizeof(uint32_t) * A->hack.size(), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(deviceX, x.data(), sizeof(uint32_t) * numWordsX, cudaMemcpyHostToDevice));

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    fprintf(stdout, "[INFO]\tTime to transfer data to device: %f ms\n", elapsedTime);

    nThreads = rows;
    blocks = (nThreads + THREAD_PER_BLOCK - 1) / THREAD_PER_BLOCK;

    dummyKernel<<<blocks, THREAD_PER_BLOCK>>>();

    /* Computational kernel execution */
    avgElapsedTime = 0.0f;

    for (int i = 0; i < LAUNCHES; ++i)
    {
        CHECK_CUDA(cudaMemset(deviceY, 0, numWordsY * sizeof(uint32_t)));

        cudaEventRecord(start, 0);
        spmvBwcHllKernel<<<blocks, THREAD_PER_BLOCK>>>(deviceWord, deviceColIdx, deviceHack, deviceX, deviceY, rows, A->hackSize);
        cudaEventRecord(stop, 0);

        CHECK_CUDA(cudaPeekAtLastError());
        CHECK_CUDA(cudaDeviceSynchronize());

        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsedTime, start, stop);
        avgElapsedTime += elapsedTime;
    }

    avgElapsedTime = avgElapsedTime / LAUNCHES;
    std::cout << "[STAT]\tComputation time: " << avgElapsedTime << "\n";

    /* Data transfer from device to host */
    cudaEventRecord(start, 0);
    CHECK_CUDA(cudaMemcpy(y.data(), deviceY, numWordsY * sizeof(uint32_t), cudaMemcpyDeviceToHost));
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    fprintf(stdout, "[INFO]\tTime to transfer data to host: %f ms\n", elapsedTime);

    /* Free device memory */
    CHECK_CUDA(cudaFree(deviceColIdx));
    CHECK_CUDA(cudaFree(deviceHack));
    CHECK_CUDA(cudaFree(deviceX));
    CHECK_CUDA(cudaFree(deviceY));
}
