cmake_minimum_required(VERSION 3.10)
project(floydwarshall_benchmark)

enable_language(CUDA)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Force CUDA compilation for .cpp source containing CUDA code
set_source_files_properties(floydwarshall.cpp PROPERTIES LANGUAGE CUDA)

add_executable(floydwarshall floydwarshall.cpp)

# CUDA compilation: target RTX 3090 (sm_86), enable optimizations
set_target_properties(floydwarshall PROPERTIES
    CUDA_ARCHITECTURES 86
)
target_compile_options(floydwarshall PRIVATE -O3 -lineinfo)

# Include common headers
target_include_directories(floydwarshall PRIVATE ../common)

# Link math library
target_link_libraries(floydwarshall m)
