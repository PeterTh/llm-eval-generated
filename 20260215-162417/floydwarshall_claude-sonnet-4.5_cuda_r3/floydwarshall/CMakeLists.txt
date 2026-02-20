cmake_minimum_required(VERSION 3.18)
project(floydwarshall_benchmark CUDA CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Enable CUDA
enable_language(CUDA)

# Find CUDA
find_package(CUDAToolkit REQUIRED)

add_executable(floydwarshall floydwarshall.cu)

# Set CUDA architectures (support common GPU architectures)
set_target_properties(floydwarshall PROPERTIES 
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "60;70;75;80;86;89;90")

# Enable optimizations for CUDA
target_compile_options(floydwarshall PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 -use_fast_math --expt-relaxed-constexpr>
    $<$<COMPILE_LANGUAGE:CXX>:-O3 -march=native -Wall -Wextra>
)

# Link math library
target_link_libraries(floydwarshall m CUDA::cudart)
