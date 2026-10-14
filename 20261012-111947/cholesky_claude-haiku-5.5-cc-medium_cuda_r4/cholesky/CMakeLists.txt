cmake_minimum_required(VERSION 3.24)
project(cholesky_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Build for the GPUs present on this machine
set(CMAKE_CUDA_ARCHITECTURES native)

# cholesky.cpp contains CUDA kernels, so compile it with nvcc
set_source_files_properties(cholesky.cpp PROPERTIES LANGUAGE CUDA)

add_executable(cholesky cholesky.cpp)

# Enable optimizations (host flags are forwarded to the host compiler under nvcc)
target_compile_options(cholesky PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-O3 -march=native -Wall -Wextra>
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 --compiler-options=-march=native,-Wall,-Wextra>)

# Link math library
target_link_libraries(cholesky m)
