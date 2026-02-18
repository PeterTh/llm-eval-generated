cmake_minimum_required(VERSION 3.10)
set(CMAKE_CUDA_ARCHITECTURES 80)
project(cholesky_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)

add_executable(cholesky cholesky.cpp)

# Enable optimizations and CUDA
target_compile_options(cholesky PRIVATE 
    -O3 -march=native -Wall -Wextra 
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 -Xcompiler -fopenmp>
)

# Link libraries
target_link_libraries(cholesky PRIVATE m MPI::MPI_CXX OpenMP::OpenMP_CXX)

# Set CUDA architecture
set_property(TARGET cholesky PROPERTY CUDA_ARCHITECTURES 80)
set_source_files_properties(cholesky.cpp PROPERTIES LANGUAGE CUDA)
