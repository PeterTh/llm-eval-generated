cmake_minimum_required(VERSION 3.10)
project(floydwarshall_benchmark)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

enable_language(CUDA)
set(CMAKE_CUDA_ARCHITECTURES 86)

find_package(MPI REQUIRED)

# Compile the .cpp file with nvcc so CUDA kernels are included
set_source_files_properties(floydwarshall.cpp PROPERTIES LANGUAGE CUDA)

add_executable(floydwarshall floydwarshall.cpp)

# Host compiler flags (passed via -Xcompiler)
set_target_properties(floydwarshall PROPERTIES
    CUDA_SEPARABLE_COMPILATION OFF
)
target_compile_options(floydwarshall PRIVATE
    -O3
    -Xcompiler=-O3
    -Xcompiler=-march=native
    -Xcompiler=-Wall
    -Xcompiler=-Wextra
    -Xcompiler=-fopenmp
)
target_compile_definitions(floydwarshall PRIVATE CUDA_STD=C++20)

# Link MPI, OpenMP, CUDA, math
target_link_libraries(floydwarshall MPI::MPI_CXX -fopenmp -lcudart -lm)

target_include_directories(floydwarshall PRIVATE ${MPI_CXX_INCLUDE_DIRS})
