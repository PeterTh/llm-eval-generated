cmake_minimum_required(VERSION 3.18)
project(cahn_hilliard_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find MPI and OpenMP
find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)

add_executable(cahn_hilliard cahn_hilliard.cpp)

# Tell CMake to compile the .cpp with NVCC (CUDA) so kernels in the same file are supported
set_source_files_properties(cahn_hilliard.cpp PROPERTIES LANGUAGE CUDA)

# Enable optimizations
if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    target_compile_options(cahn_hilliard PRIVATE -O3 -march=native -Wall -Wextra)
endif()

# Link libraries: math, MPI, OpenMP and CUDA runtime is linked automatically
target_link_libraries(cahn_hilliard PRIVATE m MPI::MPI_CXX OpenMP::OpenMP_CXX)

# Ensure CUDA uses high optimization
set_target_properties(cahn_hilliard PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(cahn_hilliard PROPERTIES CUDA_STANDARD 14)
set_target_properties(cahn_hilliard PROPERTIES CUDA_STANDARD_REQUIRED ON)
