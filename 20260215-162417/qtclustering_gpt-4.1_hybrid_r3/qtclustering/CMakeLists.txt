cmake_minimum_required(VERSION 3.10)

project(qtclustering)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)
find_package(CUDA REQUIRED)

add_executable(qtclustering qtclustering.cpp)

set_target_properties(qtclustering PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_compile_options(qtclustering PRIVATE -O3 -march=native -Wall -Wextra)
target_link_libraries(qtclustering PRIVATE MPI::MPI_CXX OpenMP::OpenMP_CXX CUDA::cudart m)
