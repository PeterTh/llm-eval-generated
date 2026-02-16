cmake_minimum_required(VERSION 3.10)
project(spmv_benchmark)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)

add_executable(spmv spmv.cpp)

# Enable optimizations
target_compile_options(spmv PRIVATE -O3 -march=native -Wall -Wextra)

# Find CUDA toolkit include path
target_include_directories(spmv PRIVATE /usr/local/cuda-12.6/targets/x86_64-linux/include)

# Link libraries
find_library(CUDA_LIB cuda HINTS /usr/local/cuda-12.6/targets/x86_64-linux/lib)
target_link_libraries(spmv m MPI::MPI_CXX OpenMP::OpenMP_CXX ${CUDA_LIB})

target_compile_definitions(spmv PRIVATE USE_MPI USE_CUDA USE_OPENMP)
