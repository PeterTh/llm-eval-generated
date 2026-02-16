cmake_minimum_required(VERSION 3.10)
cmake_minimum_required(VERSION 3.10)
project(cholesky_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

add_executable(cholesky cholesky.cpp)
set_target_properties(cholesky PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
# Set CUDA architectures for compatibility
set(CMAKE_CUDA_ARCHITECTURES 60)

# Enable optimizations
target_compile_options(cholesky PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3;--use_fast_math>)

# Link math library
target_link_libraries(cholesky m)
