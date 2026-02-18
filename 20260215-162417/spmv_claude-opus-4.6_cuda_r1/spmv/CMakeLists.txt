cmake_minimum_required(VERSION 3.10)
project(spmv_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES native)
endif()

add_executable(spmv spmv.cu)

# Enable optimizations
target_compile_options(spmv PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 --use_fast_math>
    $<$<COMPILE_LANGUAGE:CXX>:-O3 -march=native -Wall -Wextra>
)

# Link math library
target_link_libraries(spmv m)
