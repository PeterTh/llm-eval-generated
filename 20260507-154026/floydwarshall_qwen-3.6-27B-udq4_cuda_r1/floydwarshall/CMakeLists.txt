cmake_minimum_required(VERSION 3.10)
project(floydwarshall_benchmark CUDA CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

enable_language(CUDA)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Compile .cpp as CUDA to include kernel code
set_source_files_properties(floydwarshall.cpp PROPERTIES LANGUAGE CUDA)

add_executable(floydwarshall floydwarshall.cpp)

# Enable optimizations - auto-detect GPU architecture
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    execute_process(
        COMMAND nvcc --list-gpu-arch 2>/dev/null
        WORKING_DIRECTORY .
        OUTPUT_VARIABLE DETECTED_ARCH
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
    )
    if(DETECTED_ARCH)
        string(REPLACE "\n" ";" DETECTED_ARCH "${DETECTED_ARCH}")
        string(REPLACE " " "" DETECTED_ARCH "${DETECTED_ARCH}")
        set(CMAKE_CUDA_ARCHITECTURES "${DETECTED_ARCH}" CACHE STRING "GPU architectures" FORCE)
    else()
        set(CMAKE_CUDA_ARCHITECTURES 80 CACHE STRING "GPU architectures" FORCE)
    endif()
endif()
target_compile_options(floydwarshall PRIVATE -O3 -Wall -Wextra)

# Link math library
target_link_libraries(floydwarshall m)
