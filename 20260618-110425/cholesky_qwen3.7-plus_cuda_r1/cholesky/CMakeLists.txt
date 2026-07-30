cmake_minimum_required(VERSION 3.10)
project(cholesky_benchmark LANGUAGES CUDA CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Set CUDA architecture for RTX 3090 (compute capability 8.6)
set(CMAKE_CUDA_ARCHITECTURES 86)

# Compile cholesky.cpp as CUDA source
set_source_files_properties(cholesky.cpp PROPERTIES LANGUAGE CUDA)

add_executable(cholesky cholesky.cpp)

# Enable optimizations
target_compile_options(cholesky PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 --compiler-options -O3,-march=native,-Wall,-Wextra>
)

# Link math and cuSOLVER libraries
target_link_libraries(cholesky m cusolver)
