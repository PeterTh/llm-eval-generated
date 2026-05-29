cmake_minimum_required(VERSION 3.18)
project(matmul_benchmark CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find MPI and CUDA
find_package(MPI REQUIRED)
enable_language(CUDA)
set(CMAKE_CUDA_ARCHITECTURES 86)  # Ampere (RTX 30xx)

# Find OpenMP
find_package(OpenMP REQUIRED)

# Compile matmul.cpp with nvcc so CUDA kernels are included
set_source_files_properties(matmul.cpp PROPERTIES LANGUAGE CUDA)
add_executable(matmul matmul.cpp)

set_target_properties(matmul PROPERTIES LINKER_LANGUAGE CXX)

# Enable optimizations
target_compile_options(matmul PRIVATE -O3 -march=native -Wall -Wextra)

# Link MPI, OpenMP, CUDA runtime, and math
target_link_libraries(matmul MPI::MPI_CXX cudart m)
target_compile_options(matmul PRIVATE ${OpenMP_CXX_FLAGS})
target_link_libraries(matmul ${OpenMP_CXX_LIBRARIES})

# Include directories
target_include_directories(matmul PRIVATE ${MPI_CXX_INCLUDE_PATH})
