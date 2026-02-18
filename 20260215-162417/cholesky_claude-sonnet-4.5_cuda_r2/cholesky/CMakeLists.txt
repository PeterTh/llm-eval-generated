cmake_minimum_required(VERSION 3.10)
project(cholesky_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Enable CUDA
enable_language(CUDA)
find_package(CUDAToolkit REQUIRED)

add_executable(cholesky cholesky.cu)

# Set CUDA architecture (adjust as needed for your GPU)
set_target_properties(cholesky PROPERTIES CUDA_ARCHITECTURES "60;70;75;80;86;89;90")

# Enable optimizations
target_compile_options(cholesky PRIVATE 
    $<$<COMPILE_LANGUAGE:CXX>:-O3 -march=native -Wall -Wextra>
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 --use_fast_math>
)

# Link libraries
target_link_libraries(cholesky CUDA::cudart m)
