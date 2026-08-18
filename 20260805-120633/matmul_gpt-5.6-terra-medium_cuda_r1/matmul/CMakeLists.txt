cmake_minimum_required(VERSION 3.18)
project(matmul_benchmark LANGUAGES CXX CUDA)

find_package(CUDAToolkit REQUIRED)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(matmul matmul.cpp)
set_source_files_properties(matmul.cpp PROPERTIES LANGUAGE CUDA)

# This benchmark always uses the CUDA implementation.  Native architecture
# selection keeps the generated kernel tuned to the GPU on which it is built.
set_target_properties(matmul PROPERTIES
    CUDA_ARCHITECTURES native
    CUDA_SEPARABLE_COMPILATION OFF
)

# Enable optimizations.  -march=native applies only to host compilation.
target_compile_options(matmul PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-O3 -march=native -Wall -Wextra>
    $<$<COMPILE_LANGUAGE:CUDA>:-O3>
)

# cudart is linked explicitly so the target remains a CUDA-only executable.
target_link_libraries(matmul PRIVATE CUDA::cudart m)
