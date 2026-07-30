cmake_minimum_required(VERSION 3.18)
project(spmv_benchmark)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Enable CUDA support
enable_language(CUDA)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES 86)  # RTX 30xx series (Ampere)

# Find MPI
find_package(MPI REQUIRED)

# Find OpenMP
find_package(OpenMP REQUIRED)

# Mark the source file as CUDA for nvcc compilation
set_source_files_properties(spmv.cpp PROPERTIES LANGUAGE CUDA)

add_executable(spmv spmv.cpp)

# Enable optimizations
set_target_properties(spmv PROPERTIES CUDA_ARCHITECTURES "86")
target_compile_options(spmv PRIVATE -O3 -Wall -Wextra)
# Pass host compiler flags via -Xcompiler
target_compile_options(spmv PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-march=native,-fopenmp>")

# Link MPI, OpenMP, and CUDA runtime
target_link_libraries(spmv MPI::MPI_CXX OpenMP::OpenMP_CXX m)

# Include directories
target_include_directories(spmv PRIVATE ${MPI_CXX_INCLUDE_DIRS})
