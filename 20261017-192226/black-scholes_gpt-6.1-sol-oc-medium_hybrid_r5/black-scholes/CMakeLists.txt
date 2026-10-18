cmake_minimum_required(VERSION 3.18)
project(black_scholes_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(MPI REQUIRED COMPONENTS CXX)
find_package(OpenMP REQUIRED COMPONENTS CXX)
find_package(CUDAToolkit REQUIRED)

# Keep the original source and executable names; compile the source with nvcc.
set_source_files_properties(black_scholes.cpp PROPERTIES LANGUAGE CUDA)
add_executable(black_scholes black_scholes.cpp)
set_target_properties(black_scholes PROPERTIES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON)
target_compile_options(black_scholes PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3;--fmad=false;-Xcompiler=-fopenmp,-Wall,-Wextra>)

# OpenMP's imported compile options apply to CXX, so explicitly forward them
# to nvcc above as well. Retain precise double-precision CUDA math.
target_link_libraries(black_scholes PRIVATE MPI::MPI_CXX OpenMP::OpenMP_CXX CUDA::cudart m)
