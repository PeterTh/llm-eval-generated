cmake_minimum_required(VERSION 3.18)
project(black_scholes_benchmark LANGUAGES CXX CUDA)

find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Target Ampere+ GPU architectures (RTX 3090 = sm_86, A100 = sm_80, etc.)
set(CMAKE_CUDA_ARCHITECTURES "80;86")

add_executable(black_scholes black_scholes.cpp)

# Compile the source file as CUDA so nvcc handles both host and device code
set_source_files_properties(black_scholes.cpp PROPERTIES LANGUAGE CUDA)

# Enable optimisations and forward OpenMP flag through nvcc to host compiler
target_compile_options(black_scholes PRIVATE
    -O3 -march=native -Wall -Wextra
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-fopenmp>
)

target_link_libraries(black_scholes
    MPI::MPI_CXX
    OpenMP::OpenMP_CXX
    m
)
