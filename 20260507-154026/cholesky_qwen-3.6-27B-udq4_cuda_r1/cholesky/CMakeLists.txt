cmake_minimum_required(VERSION 3.10)
project(cholesky_benchmark)

enable_language(CUDA)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES OFF)

add_executable(cholesky cholesky.cpp)
set_source_files_properties(cholesky.cpp PROPERTIES LANGUAGE CUDA)

# Enable optimizations
target_compile_options(cholesky PRIVATE -O3 -Xcompiler=-march=native -Wall -Wextra)

# Link math and cuBLAS libraries
target_link_libraries(cholesky m cublas)
