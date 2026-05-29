cmake_minimum_required(VERSION 3.10)

project(qtclustering CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)

# Find MPI
find_package(MPI REQUIRED)

add_executable(qtclustering qtclustering.cpp)

set_source_files_properties(qtclustering.cpp PROPERTIES LANGUAGE CUDA)

# Pass -fopenmp and MPI flags to nvcc via -Xcompiler
target_compile_options(qtclustering PRIVATE
    -O3
    -Xcompiler=-fopenmp
    -Xcompiler=-march=native
    -Xcompiler=-Wall
    -Xcompiler=-Wextra
)

target_include_directories(qtclustering PRIVATE ${MPI_CXX_INCLUDE_DIRS})

target_link_libraries(qtclustering
    ${MPI_CXX_LIBRARIES}
    ${MPI_CXX_LINK_FLAGS}
    -fopenmp
    m
)
