cmake_minimum_required(VERSION 3.18)

# Force CUDA compilation for GPU parallelism
set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
project(stencil3d_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

add_executable(stencil3d stencil3d.cpp)
set_source_files_properties(stencil3d.cpp PROPERTIES LANGUAGE CUDA)

# Build for the local GPU for maximum performance
set(CMAKE_CUDA_ARCHITECTURES native)

find_package(CUDAToolkit REQUIRED)

# Enable optimizations
target_compile_options(stencil3d PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:-O3 -march=native -Wall -Wextra>
  $<$<COMPILE_LANGUAGE:CUDA>:-O3 -Xcompiler=-march=native>
)

# Link math + CUDA runtime
target_link_libraries(stencil3d PRIVATE m CUDA::cudart)
