cmake_minimum_required(VERSION 3.10)

project(qtclustering CUDA CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find MPI and OpenMP
find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)

add_executable(qtclustering qtclustering.cpp)

set_source_files_properties(qtclustering.cpp PROPERTIES LANGUAGE CUDA)

set_target_properties(qtclustering PROPERTIES
    CUDA_ARCHITECTURES "all"
)

target_compile_options(qtclustering PRIVATE -O3 -march=native -Xcompiler=-Wall -Xcompiler=-Wextra -Xcompiler=-fopenmp)
target_link_libraries(qtclustering PRIVATE MPI::MPI_CXX gomp m)
