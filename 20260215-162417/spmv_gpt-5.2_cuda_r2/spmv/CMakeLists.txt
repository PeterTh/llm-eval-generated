cmake_minimum_required(VERSION 3.10)

# Make CUDA compilation unconditional; nvcc is available at this path in the runtime.
if(NOT DEFINED CMAKE_CUDA_COMPILER)
    set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc" CACHE FILEPATH "CUDA compiler")
endif()

project(spmv_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

add_executable(spmv spmv.cpp)

# Compile the existing file as CUDA (no new files).
set_source_files_properties(spmv.cpp PROPERTIES LANGUAGE CUDA)

# Enable optimizations
target_compile_options(spmv PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-O3 -march=native -Wall -Wextra>
    $<$<COMPILE_LANGUAGE:CUDA>:-O3>
)

if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.18)
    set_property(TARGET spmv PROPERTY CUDA_ARCHITECTURES native)
endif()

# Link math library
target_link_libraries(spmv m)
