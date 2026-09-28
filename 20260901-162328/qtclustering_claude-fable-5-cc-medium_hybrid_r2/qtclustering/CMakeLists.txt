cmake_minimum_required(VERSION 3.18)

project(qtclustering LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES OR CMAKE_CUDA_ARCHITECTURES STREQUAL "")
    set(CMAKE_CUDA_ARCHITECTURES 86)
endif()

find_package(MPI REQUIRED COMPONENTS CXX)
find_package(OpenMP REQUIRED COMPONENTS CXX)

add_executable(qtclustering qtclustering.cpp)

# The single source file contains CUDA kernels; compile it with nvcc.
set_source_files_properties(qtclustering.cpp PROPERTIES LANGUAGE CUDA)

# -fmad=false (device) and -ffp-contract=off (host) keep CPU and GPU distance
# computations bit-identical, so results are independent of the work split.
target_compile_options(qtclustering PRIVATE
    -O3 -fmad=false
    -Xcompiler=-O3,-march=native,-fopenmp,-ffp-contract=off,-Wall,-Wextra)

# Link math library, MPI, and OpenMP runtime
target_link_libraries(qtclustering PRIVATE MPI::MPI_CXX ${OpenMP_CXX_LIBRARIES} m)
target_link_options(qtclustering PRIVATE -fopenmp)
