cmake_minimum_required(VERSION 3.24)

# This benchmark deliberately requires CUDA: the multiplication is always
# executed on the GPU and has no CPU fallback path.
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -allow-unsupported-compiler")
project(matmul_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)

add_executable(matmul matmul.cpp)
set_source_files_properties(matmul.cpp PROPERTIES LANGUAGE CUDA)

# Optimize both host code and the CUDA kernel. Native architecture selection
# lets the build use the GPU it is compiled on without hard-coding a device.
target_compile_options(matmul PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-O3;-march=native;-Wall;-Wextra>
    $<$<COMPILE_LANGUAGE:CUDA>:-O3;-Xcompiler=-Wall,-Wextra>)
set_target_properties(matmul PROPERTIES CUDA_ARCHITECTURES native CUDA_SEPARABLE_COMPILATION OFF)

# Link math library
target_link_libraries(matmul m)
