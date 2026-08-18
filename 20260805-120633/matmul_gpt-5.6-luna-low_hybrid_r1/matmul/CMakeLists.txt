cmake_minimum_required(VERSION 3.10)
project(matmul_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(matmul matmul.cpp)
find_package(MPI REQUIRED COMPONENTS CXX)
find_package(OpenMP REQUIRED)
find_package(CUDAToolkit REQUIRED)
set_source_files_properties(matmul.cpp PROPERTIES LANGUAGE CUDA)
target_link_libraries(matmul PRIVATE MPI::MPI_CXX OpenMP::OpenMP_CXX CUDA::cudart)
set_target_properties(matmul PROPERTIES CUDA_SEPARABLE_COMPILATION ON CUDA_ARCHITECTURES native)

# Enable optimizations
target_compile_options(matmul PRIVATE -O3 -march=native -Wall -Wextra $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-fopenmp>)

# Link math library
target_link_libraries(matmul PRIVATE m)
