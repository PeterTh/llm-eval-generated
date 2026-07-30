cmake_minimum_required(VERSION 3.10)
project(cholesky_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Find cuBLAS
find_package(CUDAToolkit REQUIRED)

add_executable(cholesky cholesky.cpp)

# Compile cholesky.cpp as CUDA (allows <<<>>> kernel launch syntax)
set_source_files_properties(cholesky.cpp PROPERTIES LANGUAGE CUDA)

# Enable optimizations
target_compile_options(cholesky PRIVATE -O3 -march=native -Wall -Wextra)

# Link cuBLAS and math library
target_link_libraries(cholesky CUDA::cublas m)
