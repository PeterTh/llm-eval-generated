cmake_minimum_required(VERSION 3.18)

project(qtclustering LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES "70;80;86;90")
endif()

find_package(MPI REQUIRED COMPONENTS CXX)
find_package(OpenMP REQUIRED COMPONENTS CXX)
find_package(CUDAToolkit REQUIRED)

add_executable(qtclustering qtclustering.cpp)
# The single source file contains the CUDA kernels
set_source_files_properties(qtclustering.cpp PROPERTIES LANGUAGE CUDA)

# No FMA contraction so host and device compute bit-identical distances
target_compile_options(qtclustering PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:--fmad=false -O3 -Xcompiler=-fopenmp,-O3,-march=native,-ffp-contract=off,-Wall,-Wextra>)

target_link_libraries(qtclustering PRIVATE MPI::MPI_CXX OpenMP::OpenMP_CXX CUDA::cudart m)
