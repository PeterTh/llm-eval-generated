cmake_minimum_required(VERSION 3.24)
project(spmv_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES native)
endif()
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

find_package(MPI REQUIRED COMPONENTS CXX)
find_package(OpenMP REQUIRED COMPONENTS CXX)
find_package(CUDAToolkit REQUIRED)

# Hybrid MPI + OpenMP + CUDA: compile the single source with nvcc
set_source_files_properties(spmv.cpp PROPERTIES LANGUAGE CUDA)
add_executable(spmv spmv.cpp)

# Enable optimizations (host flags forwarded to the host compiler);
# fp-contract=off keeps the multiply/add rounding of the original code
target_compile_options(spmv PRIVATE
    -O3
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-march=native,-Wall,-Wextra,-fopenmp,-ffp-contract=off>
    $<$<COMPILE_LANGUAGE:CUDA>:--fmad=false>)

# Link MPI, OpenMP, CUDA runtime and math library
target_link_libraries(spmv PRIVATE MPI::MPI_CXX OpenMP::OpenMP_CXX CUDA::cudart m)
