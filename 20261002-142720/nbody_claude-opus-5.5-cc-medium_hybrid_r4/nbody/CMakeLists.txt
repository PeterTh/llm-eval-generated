cmake_minimum_required(VERSION 3.24)

# nvcc 12.0 supports host GCC up to 12; prefer g++-12 when available.
if(NOT DEFINED CMAKE_CUDA_HOST_COMPILER AND EXISTS /usr/bin/g++-12)
    set(CMAKE_CUDA_HOST_COMPILER /usr/bin/g++-12)
endif()
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES native)
endif()

project(nbody_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(MPI REQUIRED COMPONENTS CXX)
find_package(OpenMP REQUIRED COMPONENTS CXX)
find_package(CUDAToolkit REQUIRED)

add_executable(nbody nbody.cpp)
# Hybrid MPI + OpenMP + CUDA: compile the single source as CUDA.
set_source_files_properties(nbody.cpp PROPERTIES LANGUAGE CUDA)
set_target_properties(nbody PROPERTIES LINKER_LANGUAGE CUDA)

# Enable optimizations. -fmad=false keeps device arithmetic exactly as written
# (explicit FMAs mirror the reference CPU build).
target_compile_options(nbody PRIVATE
    -O3 -fmad=false
    "SHELL:-Xcompiler -O3,-march=native,-Wall,-Wextra,-fopenmp")

target_include_directories(nbody PRIVATE ${MPI_CXX_INCLUDE_DIRS})
target_link_libraries(nbody PRIVATE MPI::MPI_CXX OpenMP::OpenMP_CXX CUDA::cudart m)
