cmake_minimum_required(VERSION 3.24)
project(cahn_hilliard_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Target the GPU(s) present at configure time unless the user overrides it
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES native)
endif()

find_package(CUDAToolkit REQUIRED)

add_executable(cahn_hilliard cahn_hilliard.cpp)

# The benchmark source is CUDA code; compile it with nvcc
set_source_files_properties(cahn_hilliard.cpp PROPERTIES LANGUAGE CUDA)

# Enable optimizations.
# -fmad=false keeps each floating-point operation separately rounded (no FMA
# contraction), so results match the original CPU operation order.
target_compile_options(cahn_hilliard PRIVATE
    -O3
    -fmad=false
    --compiler-options=-O3,-march=native,-Wall,-Wextra
)

# Link CUDA runtime and math library
target_link_libraries(cahn_hilliard PRIVATE CUDA::cudart m)
