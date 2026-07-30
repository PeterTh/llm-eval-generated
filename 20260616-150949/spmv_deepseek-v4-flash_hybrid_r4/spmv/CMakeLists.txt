cmake_minimum_required(VERSION 3.18)
project(spmv_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

add_executable(spmv spmv.cpp)

# Mark file as CUDA source so it's compiled with nvcc
set_source_files_properties(spmv.cpp PROPERTIES LANGUAGE CUDA)

# Enable optimizations
target_compile_options(spmv PRIVATE -O3 -march=native -Wall -Wextra)

# Find and link MPI
find_package(MPI REQUIRED)
target_link_libraries(spmv PUBLIC MPI::MPI_CXX)

# Find and link OpenMP
find_package(OpenMP REQUIRED)
target_link_libraries(spmv PUBLIC OpenMP::OpenMP_CXX)
# nvcc needs -Xcompiler to forward OpenMP flags to host compiler
target_compile_options(spmv PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-fopenmp>)

# Link math library
target_link_libraries(spmv PUBLIC m)
