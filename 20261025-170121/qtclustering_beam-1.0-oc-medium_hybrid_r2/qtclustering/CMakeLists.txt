cmake_minimum_required(VERSION 3.10)

project(qtclustering LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)

find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)

add_executable(qtclustering qtclustering.cpp)

target_include_directories(qtclustering PRIVATE ${MPI_CXX_INCLUDE_DIRS})
target_compile_options(qtclustering PRIVATE -O3 -march=native -Wall -Wextra)
target_compile_definitions(qtclustering PRIVATE USE_MPI USE_OPENMP USE_CUDA)

target_link_libraries(qtclustering PRIVATE MPI::MPI_CXX OpenMP::OpenMP_CXX cudart m)
