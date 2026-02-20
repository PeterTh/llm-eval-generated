cmake_minimum_required(VERSION 3.10)
project(cholesky_benchmark LANGUAGES C CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

enable_language(CUDA)

set_source_files_properties(cholesky.cpp PROPERTIES LANGUAGE CUDA)

add_executable(cholesky cholesky.cpp)

target_compile_options(cholesky PRIVATE -O3 -march=native -Wall -Wextra)
set_target_properties(cholesky PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES 60 61 70 75 80
)

target_link_libraries(cholesky m cudart)
