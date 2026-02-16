cmake_minimum_required(VERSION 3.18)
project(matmul_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)
find_package(CUDAToolkit REQUIRED)

add_executable(matmul matmul.cpp)
# Ensure the source is compiled with CUDA (nvcc) so device kernels in .cpp are accepted
set_source_files_properties(${CMAKE_CURRENT_SOURCE_DIR}/matmul.cpp PROPERTIES LANGUAGE CUDA)

# Enable optimizations and OpenMP
target_compile_options(matmul PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-O3 -march=native -Wall -Wextra>)
# NVCC flags
target_compile_options(matmul PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda -O3 --compiler-options -fopenmp>)

target_compile_definitions(matmul PRIVATE USE_MPI USE_OPENMP USE_CUDA)

# Link required libraries
target_link_libraries(matmul PRIVATE MPI::MPI_CXX OpenMP::OpenMP_CXX CUDAToolkit::cudart m)

# Ensure proper include directories for MPI and CUDA
target_include_directories(matmul PRIVATE ${MPI_CXX_INCLUDE_PATH})
