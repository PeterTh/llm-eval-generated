cmake_minimum_required(VERSION 3.18)

project(qtclustering LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_ARCHITECTURES "native")

find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)

add_executable(qtclustering qtclustering.cpp)

set_source_files_properties(qtclustering.cpp PROPERTIES LANGUAGE CUDA)

# nvcc must forward OpenMP flags to the host compiler
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler -fopenmp")

target_compile_options(qtclustering PRIVATE -O3 -march=native -Wall -Wextra)

target_link_libraries(qtclustering PRIVATE
    m
    MPI::MPI_CXX
    OpenMP::OpenMP_CXX
)
