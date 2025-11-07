cmake_minimum_required(VERSION 3.18) # CUDAToolkit requires 3.18+
project(bSpMV LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CXX_FLAGS_DEBUG "-g -O0")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -O3")

# Optional: specify the CUDA root if not automatically found
# set(CUDA_TOOLKIT_ROOT_DIR /usr/local/cuda-13.0)

# Set CUDA architectures to avoid empty error
set(CMAKE_CUDA_ARCHITECTURES "native")  # or list specific architectures

# Find CUDA using modern CMake
find_package(CUDAToolkit QUIET)

if(NOT CUDAToolkit_FOUND)
    message(WARNING "CUDA toolkit not found. GPU execution will be disabled!")
else()
    message(STATUS "CUDA toolkit found at ${CUDAToolkit_ROOT}")
endif()

add_subdirectory(src)

add_executable(bSpMV src/main.cpp)
target_link_libraries(bSpMV PRIVATE format io kernels bspmvUtils)

# If using CUDA in targets, link with CUDAToolkit::cudart
# target_link_libraries(bSpMV PRIVATE CUDAToolkit::cudart)
