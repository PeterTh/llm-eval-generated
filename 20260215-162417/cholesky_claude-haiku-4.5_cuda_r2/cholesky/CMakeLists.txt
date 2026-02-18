cmake_minimum_required(VERSION 3.17)
project(cholesky_benchmark CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Set CUDA architecture
set(CMAKE_CUDA_ARCHITECTURES 60 70 80 90)

add_executable(cholesky cholesky.cu)

# Enable optimizations
target_compile_options(cholesky PRIVATE -O3 -march=native -Wall -Wextra)
target_compile_options(cholesky PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3>)

# Set CUDA properties
set_target_properties(cholesky PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

# Link math and CUDA libraries
target_link_libraries(cholesky PRIVATE m cuda cudart cublas)
