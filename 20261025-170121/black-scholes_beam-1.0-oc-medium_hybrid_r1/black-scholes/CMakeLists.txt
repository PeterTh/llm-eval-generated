cmake_minimum_required(VERSION 3.10)
project(black_scholes_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES native)

find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)
find_package(CUDAToolkit REQUIRED)

add_executable(black_scholes black_scholes.cpp)
set_source_files_properties(black_scholes.cpp PROPERTIES LANGUAGE CUDA)

target_compile_options(black_scholes PRIVATE -O3 -Wall -Wextra -Xcompiler -fopenmp)
target_link_libraries(black_scholes PRIVATE MPI::MPI_CXX OpenMP::OpenMP_CXX CUDA::cudart m)
