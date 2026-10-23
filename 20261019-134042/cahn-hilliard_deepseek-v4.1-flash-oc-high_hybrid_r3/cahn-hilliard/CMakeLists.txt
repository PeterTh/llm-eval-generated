cmake_minimum_required(VERSION 3.18)
project(cahn_hilliard_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(MPI REQUIRED COMPONENTS CXX)
find_package(OpenMP REQUIRED COMPONENTS CXX)
find_package(CUDAToolkit REQUIRED)

add_executable(cahn_hilliard cahn_hilliard.cpp)

# The benchmark source is a single translation unit; compile it with nvcc so the
# CUDA kernels and the MPI/OpenMP host code live in the same file.
set_source_files_properties(cahn_hilliard.cpp PROPERTIES LANGUAGE CUDA)

set_target_properties(cahn_hilliard PROPERTIES
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON)

if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.24)
    set_target_properties(cahn_hilliard PROPERTIES CUDA_ARCHITECTURES native)
elseif(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set_target_properties(cahn_hilliard PROPERTIES CUDA_ARCHITECTURES "70;75;80;86")
endif()

# Enable optimizations for host and device code
target_compile_options(cahn_hilliard PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3;-Xcompiler=-march=native;-Xcompiler=-fopenmp;-Xcompiler=-Wall;-Xcompiler=-Wextra>
    $<$<COMPILE_LANGUAGE:CXX>:-O3;-march=native;-fopenmp;-Wall;-Wextra>)

# Link MPI, OpenMP, CUDA runtime, and math library
target_link_libraries(cahn_hilliard PRIVATE
    MPI::MPI_CXX
    OpenMP::OpenMP_CXX
    CUDA::cudart
    m)
