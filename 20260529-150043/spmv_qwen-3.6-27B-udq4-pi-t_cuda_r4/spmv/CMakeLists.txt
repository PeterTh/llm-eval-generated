cmake_minimum_required(VERSION 3.10)
project(spmv_benchmark)

enable_language(CUDA)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Set CUDA architecture for broad compatibility and performance
set(CMAKE_CUDA_ARCHITECTURES "50;60;61;70;75;80;86")

add_executable(spmv spmv.cpp)

# Force .cpp to be compiled as CUDA source
set_source_files_properties(spmv.cpp PROPERTIES LANGUAGE CUDA)

# Enable optimizations
target_compile_options(spmv PRIVATE -O3 --use_fast_math)
target_compile_options(spmv PRIVATE -Xcompiler=-march=native -Xcompiler=-Wall -Xcompiler=-Wextra)

# Link math library
target_link_libraries(spmv m)
