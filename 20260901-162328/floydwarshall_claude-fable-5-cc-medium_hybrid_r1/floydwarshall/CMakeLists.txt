cmake_minimum_required(VERSION 3.18)
project(floydwarshall_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(NOT CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 86)
endif()

find_package(MPI REQUIRED COMPONENTS CXX)
find_package(OpenMP REQUIRED COMPONENTS CXX)
find_package(CUDAToolkit REQUIRED)

# The single translation unit contains CUDA kernels; compile it with nvcc
set_source_files_properties(floydwarshall.cpp PROPERTIES LANGUAGE CUDA)

add_executable(floydwarshall floydwarshall.cpp)

# Enable optimizations (host-compiler flags wrapped for nvcc)
target_compile_options(floydwarshall PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 -Xcompiler=-O3,-march=native,-fopenmp,-Wall,-Wextra>)

target_include_directories(floydwarshall PRIVATE ${MPI_CXX_INCLUDE_DIRS})

set_target_properties(floydwarshall PROPERTIES LINKER_LANGUAGE CXX)

# Link math library, MPI, OpenMP, and the CUDA runtime
target_link_libraries(floydwarshall PRIVATE
    ${MPI_CXX_LIBRARIES} OpenMP::OpenMP_CXX CUDA::cudart m)
