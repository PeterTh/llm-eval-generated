cmake_minimum_required(VERSION 3.18)

project(qtclustering LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
if(NOT CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 70 80 86)
endif()

find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)
find_package(CUDAToolkit REQUIRED)

add_executable(qtclustering qtclustering.cpp)

# Compile qtclustering.cpp as CUDA source so MPI + OpenMP host code and
# CUDA device kernels can coexist in the single source file.
set_source_files_properties(qtclustering.cpp PROPERTIES LANGUAGE CUDA)

target_compile_options(qtclustering PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 --extended-lambda -Xcompiler=-fopenmp -Xcompiler=-Wall -Xcompiler=-Wextra>
)

target_include_directories(qtclustering PRIVATE ${MPI_CXX_INCLUDE_DIRS})

target_link_libraries(qtclustering PRIVATE
    m
    MPI::MPI_CXX
    OpenMP::OpenMP_CXX
    CUDA::cudart
)

set_target_properties(qtclustering PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)
