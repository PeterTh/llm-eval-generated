cmake_minimum_required(VERSION 3.18)
project(matmul_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Target the GPU present on the build machine (RTX 3090 / sm_86).
# Override with -DCMAKE_CUDA_ARCHITECTURES=<arch> when building elsewhere.
set(CMAKE_CUDA_ARCHITECTURES "86" CACHE STRING "CUDA architectures to build for")

add_executable(matmul matmul.cpp)

# The implementation lives in matmul.cpp but contains CUDA device code.
set_source_files_properties(matmul.cpp PROPERTIES LANGUAGE CUDA)

# Enable optimizations for both the host and device compilation passes
target_compile_options(matmul PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-O3 -march=native -Wall -Wextra>
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 -Xcompiler=-march=native -Wall -Wextra>
)

# Link math library
target_link_libraries(matmul m)
