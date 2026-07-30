cmake_minimum_required(VERSION 3.18)
project(cholesky_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)

# RTX 3090 = compute capability 8.6
set(CMAKE_CUDA_ARCHITECTURES "86")

# Compile cholesky.cpp as CUDA source so kernel invocations (<<<>>>) work
set_source_files_properties(cholesky.cpp PROPERTIES LANGUAGE CUDA)

# -Xcompiler=-fopenmp passes -fopenmp to the host compiler (g++) from nvcc
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler=-fopenmp -O3")

add_executable(cholesky cholesky.cpp)

target_compile_options(cholesky PRIVATE -O3 -march=native -Wall -Wextra)
target_include_directories(cholesky PRIVATE ${MPI_CXX_INCLUDE_DIRS})
target_link_libraries(cholesky PRIVATE MPI::MPI_CXX OpenMP::OpenMP_CXX m)
