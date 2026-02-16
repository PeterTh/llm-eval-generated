cmake_minimum_required(VERSION 3.10)
project(stencil3d_benchmark LANGUAGES CXX)
# Set a reasonable default CUDA architecture list to avoid CMake detection failure in some environments
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES "60;70;75" CACHE STRING "CUDA architectures" FORCE)
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)
# Enable CUDA compilation of the single source file
add_executable(stencil3d stencil3d.cpp)
# Do not force file-level CUDA language here; compile-time CUDA detection controls CUDA usage

# Enable optimizations for both host and device
target_compile_options(stencil3d PRIVATE -O3 -march=native -Wall -Wextra $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)

# Link MPI and OpenMP
target_link_libraries(stencil3d PRIVATE MPI::MPI_CXX OpenMP::OpenMP_CXX m)
