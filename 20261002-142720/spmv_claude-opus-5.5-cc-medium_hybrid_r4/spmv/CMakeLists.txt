cmake_minimum_required(VERSION 3.18)
project(spmv_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.24)
        set(CMAKE_CUDA_ARCHITECTURES native)
    else()
        set(CMAKE_CUDA_ARCHITECTURES 70 80 86)
    endif()
endif()

find_package(MPI REQUIRED COMPONENTS CXX)
find_package(OpenMP REQUIRED COMPONENTS CXX)
find_package(CUDAToolkit REQUIRED)

# Hybrid MPI + OpenMP + CUDA build: the single source is compiled by nvcc
set_source_files_properties(spmv.cpp PROPERTIES LANGUAGE CUDA)
add_executable(spmv spmv.cpp)

# Enable optimizations
target_compile_options(spmv PRIVATE
    -O3
    -Xcompiler=-O3,-march=native,-Wall,-Wextra,${OpenMP_CXX_FLAGS})
target_include_directories(spmv PRIVATE ${MPI_CXX_INCLUDE_DIRS})
target_compile_definitions(spmv PRIVATE OMPI_SKIP_MPICXX MPICH_SKIP_MPICXX)

# Link MPI, OpenMP, CUDA runtime and math library
target_link_libraries(spmv PRIVATE ${MPI_CXX_LIBRARIES} OpenMP::OpenMP_CXX CUDA::cudart m)
