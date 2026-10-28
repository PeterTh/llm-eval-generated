cmake_minimum_required(VERSION 3.24)
project(cholesky_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Target the GPU(s) present at configure time
set(CMAKE_CUDA_ARCHITECTURES native)

# cholesky.cpp contains CUDA kernels, so it is compiled by nvcc
set_source_files_properties(cholesky.cpp PROPERTIES LANGUAGE CUDA)

add_executable(cholesky cholesky.cpp)

# Enable optimizations (host code is passed through nvcc to the host compiler)
target_compile_options(cholesky PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-O3 -march=native -Wall -Wextra>
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 -Xcompiler=-march=native,-Wall,-Wextra>
)

# Link math library
target_link_libraries(cholesky m)
