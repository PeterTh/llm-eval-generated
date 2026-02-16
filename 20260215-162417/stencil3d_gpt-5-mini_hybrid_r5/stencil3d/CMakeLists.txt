cmake_minimum_required(VERSION 3.18)
project(stencil3d_benchmark LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(stencil3d stencil3d.cpp)

# Find MPI and OpenMP; CUDA is optional
find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)
find_package(CUDAToolkit QUIET)

# Enable optimizations
target_compile_options(stencil3d PRIVATE -O3 -march=native -Wall -Wextra)

# Link libraries (link CUDA runtime if found)
if(CUDAToolkit_FOUND)
  target_link_libraries(stencil3d PRIVATE MPI::MPI_CXX OpenMP::OpenMP_CXX CUDA::cudart m)
else()
  target_link_libraries(stencil3d PRIVATE MPI::MPI_CXX OpenMP::OpenMP_CXX m)
endif()

# Include MPI headers if needed
if(MPI_CXX_INCLUDE_PATH)
  target_include_directories(stencil3d PRIVATE ${MPI_CXX_INCLUDE_PATH})
endif()
