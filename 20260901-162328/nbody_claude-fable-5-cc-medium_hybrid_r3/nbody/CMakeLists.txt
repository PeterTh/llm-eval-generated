cmake_minimum_required(VERSION 3.18)

# CUDA 12.0 does not support GCC 13 as host compiler; fall back to gcc-12 if needed
if(NOT DEFINED CMAKE_CUDA_HOST_COMPILER AND NOT DEFINED ENV{CUDAHOSTCXX})
    find_program(GXX12 g++-12)
    if(GXX12)
        set(CMAKE_CUDA_HOST_COMPILER ${GXX12})
    endif()
endif()

project(nbody_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES native)
endif()

find_package(MPI REQUIRED COMPONENTS CXX)
find_package(OpenMP REQUIRED COMPONENTS CXX)

add_executable(nbody nbody.cpp)

# The whole program (host + device code) lives in nbody.cpp; compile it as CUDA
set_source_files_properties(nbody.cpp PROPERTIES LANGUAGE CUDA)

# Enable optimizations (host-compiler flags forwarded through nvcc)
target_compile_options(nbody PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 -Xcompiler=-O3,-march=native,-fopenmp,-Wall,-Wextra>)

target_include_directories(nbody PRIVATE ${MPI_CXX_INCLUDE_DIRS})

# Link MPI, OpenMP runtime, and math library
target_link_libraries(nbody PRIVATE ${MPI_CXX_LIBRARIES} OpenMP::OpenMP_CXX m)
