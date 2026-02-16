cmake_minimum_required(VERSION 3.18)

# Set a default CUDA architecture to allow detection when CUDA's default isn't available
# Placed before project() so CMake uses it for CUDA detection
set(CMAKE_CUDA_ARCHITECTURES 60)

project(floydwarshall_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Compile the existing source with NVCC by forcing CUDA language property
set(SRC ${CMAKE_CURRENT_SOURCE_DIR}/floydwarshall.cpp)
set_source_files_properties(${SRC} PROPERTIES LANGUAGE CUDA)
add_executable(floydwarshall ${SRC})

# Enable optimizations for host and device
target_compile_options(floydwarshall PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-O3;-march=native;-Wall;-Wextra>
    $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math;-O3>
)

# Find and link CUDA runtime
find_package(CUDAToolkit REQUIRED)

# Link math and CUDA runtime
target_link_libraries(floydwarshall PRIVATE m CUDAToolkit::cudart)
