cmake_minimum_required(VERSION 3.18)

# Set CUDA architectures before project
set(CMAKE_CUDA_ARCHITECTURES "60;70;75;80;86;89;90")

project(cholesky_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Find CUDA
find_package(CUDAToolkit REQUIRED)

add_executable(cholesky cholesky.cu)

# Enable optimizations
target_compile_options(cholesky PRIVATE 
    $<$<COMPILE_LANGUAGE:CXX>:-O3 -march=native -Wall -Wextra>
    $<$<COMPILE_LANGUAGE:CUDA>:-O3>
)

set_target_properties(cholesky PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# Link math library and CUDA runtime
target_link_libraries(cholesky m CUDA::cudart)
