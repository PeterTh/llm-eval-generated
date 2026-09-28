cmake_minimum_required(VERSION 3.18)
project(floydwarshall_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)
find_package(CUDAToolkit REQUIRED)

# The benchmark is a hybrid MPI + OpenMP + CUDA program; the single source
# file contains device code and therefore is compiled by the CUDA compiler.
set_source_files_properties(floydwarshall.cpp PROPERTIES LANGUAGE CUDA)

add_executable(floydwarshall floydwarshall.cpp)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set_target_properties(floydwarshall PROPERTIES CUDA_ARCHITECTURES "70;80;86;89;90")
endif()

# Enable optimizations
target_compile_options(floydwarshall PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 --expt-relaxed-constexpr -Xcompiler=-O3,-march=native,-Wall,-Wextra,-fopenmp>)

target_link_libraries(floydwarshall PRIVATE MPI::MPI_CXX OpenMP::OpenMP_CXX CUDA::cudart m)
