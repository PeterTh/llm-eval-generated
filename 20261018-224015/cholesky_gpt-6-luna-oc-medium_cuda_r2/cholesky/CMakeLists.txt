cmake_minimum_required(VERSION 3.18)
project(cholesky_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(cholesky cholesky.cu)

# Enable optimizations
target_compile_options(cholesky PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3>
)
set_target_properties(cholesky PROPERTIES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON)

# Link math library
target_link_libraries(cholesky m)
