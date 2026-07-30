cmake_minimum_required(VERSION 3.10)
project(matmul_benchmark CUDA CXX)

enable_language(CUDA)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Detect GPU architecture from the current device
if(NOT CMAKE_CUDA_ARCHITECTURES)
    execute_process(
        COMMAND bash -c "nvidia-smi --query-gpu=compute_cap --format=csv,noheader -i 0 2>/dev/null | tr -d ' ' || echo ''"
        OUTPUT_VARIABLE GPU_ARCH
        ERROR_QUIET
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    if(NOT GPU_ARCH STREQUAL "")
        # Convert "8.6" to "86"
        string(REPLACE "." "" GPU_ARCH "${GPU_ARCH}")
        set(CMAKE_CUDA_ARCHITECTURES ${GPU_ARCH})
    else()
        # Fallback: common architectures
        set(CMAKE_CUDA_ARCHITECTURES 75 80 86 89)
    endif()
endif()

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

add_executable(matmul matmul.cpp)

# Treat the .cpp source as CUDA (compiled with nvcc)
set_source_files_properties(matmul.cpp PROPERTIES LANGUAGE CUDA)

# Host compiler optimization flags
set(CMAKE_CUDA_HOST_COMPILER ${CMAKE_CXX_COMPILER})

# Enable optimizations
target_compile_options(matmul PRIVATE -O3 -Wall -Wextra)
target_compile_options(matmul PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:-O3>")

# Link math library
target_link_libraries(matmul m)

# Include common headers
target_include_directories(matmul PRIVATE ../common)
