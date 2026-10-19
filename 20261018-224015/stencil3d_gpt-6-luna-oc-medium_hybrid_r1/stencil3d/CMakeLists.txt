cmake_minimum_required(VERSION 3.10)
project(stencil3d_benchmark LANGUAGES CXX CUDA)
if(POLICY CMP0104)
  cmake_policy(SET CMP0104 NEW)
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(stencil3d stencil3d.cpp)

find_package(MPI REQUIRED COMPONENTS CXX)
find_package(OpenMP REQUIRED COMPONENTS CXX)
set_source_files_properties(stencil3d.cpp PROPERTIES LANGUAGE CUDA)
set_target_properties(stencil3d PROPERTIES CUDA_STANDARD 20 CUDA_STANDARD_REQUIRED ON)
if(NOT CMAKE_CUDA_ARCHITECTURES)
  set_property(TARGET stencil3d PROPERTY CUDA_ARCHITECTURES 60)
endif()
target_link_libraries(stencil3d PRIVATE MPI::MPI_CXX OpenMP::OpenMP_CXX)

# Enable optimizations
target_compile_options(stencil3d PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-O3;-march=native;-Wall;-Wextra> $<$<COMPILE_LANGUAGE:CUDA>:-O3;-Xcompiler=-fopenmp>)

# Link math library
target_link_libraries(stencil3d PRIVATE m)
