cmake_minimum_required(VERSION 3.24)
project(roomsim_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Build for the GPUs present on the accelerator node unless the user selected
# explicit architectures (CMake's generic default is 52).
if(CMAKE_CUDA_ARCHITECTURES STREQUAL "" OR CMAKE_CUDA_ARCHITECTURES STREQUAL "52")
    set(CMAKE_CUDA_ARCHITECTURES native)
endif()

find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)
find_package(CUDAToolkit REQUIRED)

# Compile the single translation unit with the CUDA compiler so that the
# MPI + OpenMP host code and the CUDA kernels live in the same file.
set_source_files_properties(roomsim.cpp PROPERTIES LANGUAGE CUDA)

add_executable(roomsim roomsim.cpp)

# Host code: optimized native build with OpenMP. Device code: optimized
# single-precision math (no fast-math, to preserve numerics).
target_compile_options(roomsim PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 -Xcompiler=-O3,-march=native,-fopenmp>
    $<$<COMPILE_LANGUAGE:CXX>:-O3 -march=native -fopenmp -Wall -Wextra -Wpedantic>)

target_link_libraries(roomsim PRIVATE
    MPI::MPI_CXX
    CUDA::cudart
    ${OpenMP_CXX_LIBRARIES}
    m)
