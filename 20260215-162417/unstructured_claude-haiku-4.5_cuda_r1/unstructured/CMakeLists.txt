cmake_minimum_required(VERSION 3.10)

# Set CUDA toolkit location
set(CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda" CACHE PATH "CUDA toolkit root directory")
set(CMAKE_CUDA_COMPILER "${CUDA_TOOLKIT_ROOT_DIR}/bin/nvcc" CACHE FILEPATH "CUDA compiler")

project(unstructured_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

add_executable(unstructured unstructured.cu)

# Enable optimizations for CUDA
set_target_properties(unstructured PROPERTIES
    CUDA_ARCHITECTURES 70;75;80;86;90
)

target_compile_options(unstructured PRIVATE 
    $<$<COMPILE_LANGUAGE:CXX>:-O3 -march=native -Wall -Wextra>
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 -use_fast_math>
)

# Link math library
target_link_libraries(unstructured m)
