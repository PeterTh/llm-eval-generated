cmake_minimum_required(VERSION 3.18)

project(qtclustering LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES 70 75 80 86 89 90)

find_package(MPI REQUIRED)

add_executable(qtclustering qtclustering.cpp)
set_source_files_properties(qtclustering.cpp PROPERTIES LANGUAGE CUDA)

target_include_directories(qtclustering PRIVATE ${MPI_CXX_INCLUDE_DIRS})

target_compile_options(qtclustering PRIVATE
    -O3
    -Xcompiler=-fopenmp
    -Xcompiler=-O3
    -Xcompiler=-march=native
    -Xcompiler=-Wall
    -Xcompiler=-Wextra)

target_link_libraries(qtclustering PRIVATE ${MPI_CXX_LIBRARIES} gomp m)
