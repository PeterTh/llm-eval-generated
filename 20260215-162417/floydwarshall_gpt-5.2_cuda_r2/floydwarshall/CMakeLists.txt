cmake_minimum_required(VERSION 3.10)

# Ensure NVCC is found even when /usr/local/cuda/bin isn't on PATH.
if (NOT DEFINED CMAKE_CUDA_COMPILER AND EXISTS "/usr/local/cuda/bin/nvcc")
    set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc" CACHE FILEPATH "CUDA compiler" FORCE)
endif()

# Avoid toolchain auto-detection failures on systems where "native" GPU probing isn't available.
set(CMAKE_CUDA_ARCHITECTURES 70 75 80 86)

project(floydwarshall_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

add_executable(floydwarshall floydwarshall.cpp)

# Compile the existing .cpp with NVCC to enable CUDA kernels.
set_source_files_properties(floydwarshall.cpp PROPERTIES LANGUAGE CUDA)

# Enable optimizations
target_compile_options(floydwarshall PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-O3 -march=native -Wall -Wextra>
    $<$<COMPILE_LANGUAGE:CUDA>:-O3>
)

# Link math library
target_link_libraries(floydwarshall m)
