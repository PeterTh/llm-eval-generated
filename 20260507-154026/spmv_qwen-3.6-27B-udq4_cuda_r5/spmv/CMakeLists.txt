cmake_minimum_required(VERSION 3.10)
project(spmv_benchmark CUDA CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(spmv spmv.cu)

# Enable optimizations
set_target_properties(spmv PROPERTIES CUDA_ARCHITECTURES "all")
target_compile_options(spmv PRIVATE -O3 -Xcompiler=-march=native,-Wall,-Wextra)

# Link math library
target_link_libraries(spmv m)
