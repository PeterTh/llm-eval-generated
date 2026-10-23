cmake_minimum_required(VERSION 3.24)
project(floydwarshall_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Build for the GPU(s) present on the build machine unless the user overrides it.
if(NOT CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES native)
endif()

add_executable(floydwarshall floydwarshall.cpp)

# Compile the benchmark translation unit with the CUDA compiler so the
# parallel kernels and the host benchmark live in the same source file.
set_source_files_properties(floydwarshall.cpp PROPERTIES LANGUAGE CUDA)
set_target_properties(floydwarshall PROPERTIES CUDA_SEPARABLE_COMPILATION OFF)

# Enable optimizations (host and device)
target_compile_options(floydwarshall PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-O3;-march=native;-Wall;-Wextra>
    $<$<COMPILE_LANGUAGE:CUDA>:-O3;-Wall;-Wextra>)

# Link math library
target_link_libraries(floydwarshall m)
