cmake_minimum_required(VERSION 3.10)
project(matmul_benchmark)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Enable CUDA support
enable_language(CUDA)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Detect available CUDA architectures
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    execute_process(
        COMMAND nvcc --list-gpu-archs
        OUTPUT_VARIABLE CUDA_ARCH_OUTPUT
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
        RESULT_VARIABLE CUDA_ARCH_RESULT
    )
    if(CUDA_ARCH_RESULT EQUAL 0 AND CUDA_ARCH_OUTPUT)
        set(CMAKE_CUDA_ARCHITECTURES "${CUDA_ARCH_OUTPUT}")
    else()
        # Fallback: common architectures
        set(CMAKE_CUDA_ARCHITECTURES 70;75;80;86)
    endif()
endif()

add_executable(matmul matmul.cpp)
set_source_files_properties(matmul.cpp PROPERTIES LANGUAGE CUDA)

# Enable optimizations
target_compile_options(matmul PRIVATE -O3 -Wall -Wextra)
target_compile_options(matmul PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>")

# Link math library
target_link_libraries(matmul m)
