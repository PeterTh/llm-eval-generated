cmake_minimum_required(VERSION 3.18)
project(matmul_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Target GPU architectures (Volta through Hopper). Override with -DMATMUL_CUDA_ARCHITECTURES=...
set(MATMUL_CUDA_ARCHITECTURES "70;75;80;86;89;90" CACHE STRING "CUDA architectures to build for")
set(CMAKE_CUDA_ARCHITECTURES ${MATMUL_CUDA_ARCHITECTURES})

find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)
find_package(CUDAToolkit REQUIRED)

# The hybrid kernel contains device code, so the source is compiled by nvcc.
# Host code is forwarded to the host compiler with OpenMP and native tuning.
set_source_files_properties(matmul.cpp PROPERTIES LANGUAGE CUDA)

add_executable(matmul matmul.cpp)

# -ffp-contract=off and --fmad=false keep multiply/add rounding explicit so that results
# match the reference rounding sequence bit for bit (see matmul.cpp).
target_compile_options(matmul PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 --fmad=false -Xcompiler=-O3,-march=native,-fopenmp,-ffp-contract=off,-Wall,-Wextra>
)

target_link_libraries(matmul PRIVATE
    MPI::MPI_CXX
    CUDA::cudart
    m
)
target_link_options(matmul PRIVATE -fopenmp)
