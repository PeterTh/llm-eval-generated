cmake_minimum_required(VERSION 3.18)
project(roomsim_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# GPU architectures: Volta through Hopper
set(CMAKE_CUDA_ARCHITECTURES 70 75 80 86 89 90)

# Hybrid parallelization is mandatory: MPI (ranks), OpenMP (host threads), CUDA (GPUs)
find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)
find_package(CUDAToolkit REQUIRED)

add_executable(roomsim roomsim.cpp)

# The translation unit contains device kernels, so it is compiled by nvcc
set_source_files_properties(roomsim.cpp PROPERTIES LANGUAGE CUDA)

# Enable optimizations and warnings (host flags are forwarded through nvcc;
# -Wpedantic is omitted because nvcc's #line directives trigger it spuriously).
# --fmad=false keeps GPU floating-point evaluation free of fused multiply-add
# contraction so results match the host arithmetic.
target_compile_options(roomsim PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3>
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
    $<$<COMPILE_LANGUAGE:CUDA>:--fmad=false>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-march=native>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wextra>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-fopenmp>
)
target_link_options(roomsim PRIVATE -fopenmp)

# Link MPI, OpenMP, CUDA runtime, and math library
target_link_libraries(roomsim PRIVATE MPI::MPI_CXX CUDA::cudart m)
