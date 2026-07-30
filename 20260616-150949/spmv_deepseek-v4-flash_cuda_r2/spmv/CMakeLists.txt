cmake_minimum_required(VERSION 3.18)
project(spmv_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(spmv spmv.cpp)
set_source_files_properties(spmv.cpp PROPERTIES LANGUAGE CUDA)

# Set CUDA architecture for RTX 3090 (compute capability 8.6)
set_target_properties(spmv PROPERTIES CUDA_ARCHITECTURES "86")

# Enable optimizations
target_compile_options(spmv PRIVATE -O3 -Wall -Wextra)
target_compile_options(spmv PRIVATE "$<$<COMPILE_LANGUAGE:CXX>:-march=native>")
target_compile_options(spmv PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-march=native>")

# Link math library
target_link_libraries(spmv m)
