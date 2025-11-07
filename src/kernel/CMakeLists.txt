# src/kernel/CMakeLists.txt

set(KERNEL_SOURCES
    MatrixKernel.cpp
    cpu/MatrixKernelCPU.cpp
    gpu/MatrixKernelGPU.cu
    gpu/coo.cu
    gpu/csr.cu
    gpu/ell.cu
    gpu/hll.cu
    gpu/bwcCoo.cu
    gpu/bwcCsr.cu
    gpu/bwcEll.cu
    gpu/bwcHll.cu
)

add_library(kernels ${KERNEL_SOURCES})

# Tell NVCC explicitly to use C++17
target_compile_options(kernels PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--std=c++17>)

# Still needed for separable compilation
set_target_properties(kernels PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

target_include_directories(kernels PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(kernels PUBLIC format)
