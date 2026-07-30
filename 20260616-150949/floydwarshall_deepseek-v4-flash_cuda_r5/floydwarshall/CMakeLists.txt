cmake_minimum_required(VERSION 3.18)
project(floydwarshall_benchmark)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Enable CUDA
enable_language(CUDA)

# Set CUDA architecture for RTX 3090 (compute capability 8.6)
set(CMAKE_CUDA_ARCHITECTURES "86")

add_executable(floydwarshall floydwarshall.cpp)

# Set CUDA standard
set_target_properties(floydwarshall PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON
)

# Compile as CUDA source
set_source_files_properties(floydwarshall.cpp PROPERTIES LANGUAGE CUDA)

# Enable optimizations (passed through nvcc to both host and device compiler)
target_compile_options(floydwarshall PRIVATE -O3 -march=native -Wall -Wextra)

# Link math library
target_link_libraries(floydwarshall m)
