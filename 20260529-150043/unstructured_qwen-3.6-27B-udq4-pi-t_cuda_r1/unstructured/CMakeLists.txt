cmake_minimum_required(VERSION 3.10)
project(unstructured_benchmark)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Enable CUDA language
enable_language(CUDA)

# Auto-detect CUDA architectures if not specified
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 80 CACHE STRING "CUDA architectures" FORCE)
endif()

add_executable(unstructured unstructured.cpp)

# Force CUDA compilation for the .cpp file (it contains CUDA kernels)
set_source_files_properties(unstructured.cpp PROPERTIES LANGUAGE CUDA)

# Set CUDA properties
set_target_properties(unstructured PROPERTIES
    CUDA_STANDARD 20
    CUDA_STANDARD_REQUIRED ON
)

# Enable optimizations
target_compile_options(unstructured PRIVATE -O3 -march=native -Wall -Wextra)
target_compile_options(unstructured PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:-O3;-expt-relaxed-constexpr>")

# Link CUDA runtime and math library
target_link_libraries(unstructured cudart m)
