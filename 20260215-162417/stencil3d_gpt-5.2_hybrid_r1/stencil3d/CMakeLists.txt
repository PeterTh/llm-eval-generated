cmake_minimum_required(VERSION 3.10)
project(stencil3d_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Make CUDA builds robust on systems where CMake can't auto-detect a GPU.
# Adjust to match your cluster GPUs for best performance.
set(CMAKE_CUDA_ARCHITECTURES 70 80 90 CACHE STRING "CUDA architectures" FORCE)

find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)
find_package(CUDAToolkit REQUIRED)

add_executable(stencil3d stencil3d.cpp)
set_source_files_properties(stencil3d.cpp PROPERTIES LANGUAGE CUDA)

# Enable optimizations
target_compile_options(stencil3d PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:-O3 -march=native -Wall -Wextra>
  $<$<COMPILE_LANGUAGE:CUDA>:-O3 -Xcompiler=-fopenmp>
)

target_link_libraries(stencil3d PRIVATE MPI::MPI_CXX OpenMP::OpenMP_CXX CUDA::cudart m)
