cmake_minimum_required(VERSION 3.10)
project(matmul_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Treat the single source file as CUDA so device kernels can live in it
set_source_files_properties(matmul.cpp PROPERTIES LANGUAGE CUDA)

find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)
find_package(CUDAToolkit REQUIRED)

add_executable(matmul matmul.cpp)

# Enable optimizations
target_compile_options(matmul PRIVATE -O3 -march=native -Wall -Wextra)
# Let OpenMP and CUDA compile/link flags be applied automatically
target_link_libraries(matmul PRIVATE MPI::MPI_CXX OpenMP::OpenMP_CXX CUDA::cudart m)

# Ensure position independent code for CUDA linking in some environments
set_target_properties(matmul PROPERTIES POSITION_INDEPENDENT_CODE ON)
