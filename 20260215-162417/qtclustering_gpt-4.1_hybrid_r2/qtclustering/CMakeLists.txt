cmake_minimum_required(VERSION 3.10)

project(qtclustering)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)
set(CMAKE_CUDA_ARCHITECTURES 60 70 75 80 86)
enable_language(CUDA)

add_executable(qtclustering qtclustering.cpp)
target_include_directories(qtclustering PRIVATE /usr/local/cuda/include)

target_compile_options(qtclustering PRIVATE -O3 -march=native -Wall -Wextra $<$<COMPILE_LANGUAGE:CXX>:-fopenmp>)

# Link math, MPI, OpenMP, and CUDA libraries
target_link_libraries(qtclustering m MPI::MPI_CXX OpenMP::OpenMP_CXX cuda)
