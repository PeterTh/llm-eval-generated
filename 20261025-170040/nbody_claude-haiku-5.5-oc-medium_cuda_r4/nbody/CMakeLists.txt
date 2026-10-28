cmake_minimum_required(VERSION 3.24)
project(nbody_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Build for the GPUs present on this machine (override with -DCMAKE_CUDA_ARCHITECTURES=...)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES native)
endif()

# nbody.cpp contains CUDA kernels, so it is compiled by nvcc
set_source_files_properties(nbody.cpp PROPERTIES LANGUAGE CUDA)

add_executable(nbody nbody.cpp)

# Enable optimizations. Host-side flags are forwarded to the host compiler via -Xcompiler.
target_compile_options(nbody PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 -Xcompiler=-march=native -Xcompiler=-Wall -Xcompiler=-Wextra>
)

# Link math library
target_link_libraries(nbody m)
