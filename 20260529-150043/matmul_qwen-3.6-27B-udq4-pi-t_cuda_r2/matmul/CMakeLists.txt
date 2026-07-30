cmake_minimum_required(VERSION 3.10)
project(matmul_benchmark)

enable_language(CUDA)

# Target common GPU architectures for maximum compatibility and performance
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 50 52 60 61 70 75 80 86 89 90)
endif()

add_executable(matmul matmul.cpp)

# Force nvcc to compile the .cpp file (it contains CUDA kernel code)
set_source_files_properties(matmul.cpp PROPERTIES LANGUAGE CUDA)

set_target_properties(matmul PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# Enable optimizations
target_compile_options(matmul PRIVATE
    -O3 -Wall -Wextra
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda>
)

# Link math library
target_link_libraries(matmul m)
