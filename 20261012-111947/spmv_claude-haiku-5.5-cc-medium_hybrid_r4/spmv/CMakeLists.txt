cmake_minimum_required(VERSION 3.24)
project(spmv_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Target the GPUs present on the build node
set(CMAKE_CUDA_ARCHITECTURES native)

find_package(MPI REQUIRED COMPONENTS CXX)
find_package(OpenMP REQUIRED COMPONENTS CXX)

# The MPI+OpenMP+CUDA code is a single translation unit; compile it with nvcc
set_source_files_properties(spmv.cpp PROPERTIES LANGUAGE CUDA)

add_executable(spmv spmv.cpp)

# Enable optimizations; host-side flags are forwarded to the host compiler by nvcc
target_compile_options(spmv PRIVATE -O3 -Xcompiler=-march=native -Xcompiler=-Wall -Xcompiler=-Wextra)

# OpenMP for host-side parallel work (reference, SELL build)
target_compile_options(spmv PRIVATE -Xcompiler ${OpenMP_CXX_FLAGS})
target_link_options(spmv PRIVATE ${OpenMP_CXX_FLAGS})

# Link MPI and the math library
target_link_libraries(spmv PRIVATE MPI::MPI_CXX m)
