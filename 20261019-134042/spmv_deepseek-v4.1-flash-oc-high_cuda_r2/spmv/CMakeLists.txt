cmake_minimum_required(VERSION 3.18)

# Compile for the architecture(s) of the GPU(s) present on the build machine.
# This must happen before project() so the compiler detection does not fall
# back to the toolchain default while native detection is available.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES AND CMAKE_VERSION VERSION_GREATER_EQUAL "3.24")
    set(CMAKE_CUDA_ARCHITECTURES native)
endif()

project(spmv_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

add_executable(spmv spmv.cpp)

# The benchmark source contains CUDA device code; compile it with nvcc.
set_source_files_properties(spmv.cpp PROPERTIES LANGUAGE CUDA)

# Enable optimizations
target_compile_options(spmv PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-O3 -march=native -Wall -Wextra>
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 -Xcompiler -Wall -Xcompiler -Wextra>)

# Link math library
target_link_libraries(spmv m)
