cmake_minimum_required(VERSION 3.18)

project(qtclustering LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find MPI and OpenMP
find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)

add_executable(qtclustering qtclustering.cpp)

# Compile the source file as CUDA
set_source_files_properties(qtclustering.cpp PROPERTIES LANGUAGE CUDA)

# Compile options
target_compile_options(qtclustering PRIVATE -O3 -march=native -Wall -Wextra)
target_compile_options(qtclustering PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-fopenmp>)

# Link libraries
target_link_libraries(qtclustering PRIVATE
    m
    MPI::MPI_CXX
    OpenMP::OpenMP_CXX
    ${CUDA_LIBRARIES}
)

# Set CUDA architecture for common GPU types
set_target_properties(qtclustering PROPERTIES
    CUDA_ARCHITECTURES "70;75;80;86;90"
)
