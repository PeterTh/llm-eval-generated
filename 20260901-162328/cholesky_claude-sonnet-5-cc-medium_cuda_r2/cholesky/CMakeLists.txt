cmake_minimum_required(VERSION 3.18)
project(cholesky_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(NOT CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES native)
endif()

# Compile the (CUDA-enabled) source as CUDA despite its .cpp extension.
set_source_files_properties(cholesky.cpp PROPERTIES LANGUAGE CUDA)

add_executable(cholesky cholesky.cpp)

# Enable optimizations
target_compile_options(cholesky PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3 -Xcompiler=-O3,-Wall,-Wextra>)

set_target_properties(cholesky PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# Link math library
target_link_libraries(cholesky m)
