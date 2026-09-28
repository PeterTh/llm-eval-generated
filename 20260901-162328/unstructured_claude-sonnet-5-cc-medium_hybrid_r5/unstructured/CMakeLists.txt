cmake_minimum_required(VERSION 3.18)
project(unstructured_benchmark CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)
find_package(CUDAToolkit REQUIRED)

add_executable(unstructured unstructured.cpp)

# Enable optimizations
target_compile_options(unstructured PRIVATE -O3 -march=native -Wall -Wextra)

target_include_directories(unstructured PRIVATE ${CUDAToolkit_INCLUDE_DIRS})

# Hybrid MPI (distributed decomposition) + OpenMP (host-side parallelism)
# + CUDA (per-rank GPU offload via the driver API / NVRTC) parallelization.
target_link_libraries(unstructured PRIVATE
    MPI::MPI_CXX
    OpenMP::OpenMP_CXX
    CUDA::cuda_driver
    CUDA::nvrtc
    m
)
