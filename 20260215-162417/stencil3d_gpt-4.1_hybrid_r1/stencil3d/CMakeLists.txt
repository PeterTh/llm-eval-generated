cmake_minimum_required(VERSION 3.10)
project(stencil3d_benchmark)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

enable_language(CUDA)
find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)

add_executable(stencil3d stencil3d.cpp)
set_target_properties(stencil3d PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# Enable optimizations
target_compile_options(stencil3d PRIVATE -O3 -march=native -Wall -Wextra $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-O3>)

# Link math library, MPI, OpenMP, CUDA
if(OpenMP_CXX_FOUND)
  target_link_libraries(stencil3d PUBLIC OpenMP::OpenMP_CXX)
endif()
target_link_libraries(stencil3d PUBLIC MPI::MPI_CXX m cuda)
