cmake_minimum_required(VERSION 3.10)
project(stencil3d_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

add_executable(stencil3d stencil3d.cpp)

# Tell CMake to compile the .cpp file with NVCC (CUDA) so CUDA kernels in the file are handled
set_source_files_properties(stencil3d.cpp PROPERTIES LANGUAGE CUDA)

# Enable optimizations
# For C++
target_compile_options(stencil3d PRIVATE -O3 -march=native -Wall -Wextra)
# For CUDA
target_compile_options(stencil3d PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3 --use_fast_math>)

# Link math and CUDA runtime libraries
target_link_libraries(stencil3d PRIVATE m cudart)
