cmake_minimum_required(VERSION 3.10)
project(spmv_benchmark)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find MPI
find_package(MPI REQUIRED)

# Find CUDA (legacy module for nvcc integration)
find_package(CUDA REQUIRED)

# Enable CUDA language for modern CMake integration
enable_language(CUDA)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Create executable — mark .cpp as CUDA source so nvcc compiles it
add_executable(spmv spmv.cpp)
set_source_files_properties(spmv.cpp PROPERTIES LANGUAGE CUDA)

# Set CUDA architecture
set_target_properties(spmv PROPERTIES CUDA_ARCHITECTURES "70")

# Compile options: host optimizations + OpenMP via -Xcompiler
target_compile_options(spmv PRIVATE -O3 -march=native -Wall -Wextra -Xcompiler=-fopenmp)

# Include directories
target_include_directories(spmv PRIVATE ${MPI_CXX_INCLUDE_DIRS} ${CUDA_INCLUDE_DIRS})

# Link MPI, CUDA runtime, OpenMP, and math
target_link_libraries(spmv ${MPI_CXX_LIBRARIES} ${CUDA_cudart_LIBRARY} m -lgomp)
