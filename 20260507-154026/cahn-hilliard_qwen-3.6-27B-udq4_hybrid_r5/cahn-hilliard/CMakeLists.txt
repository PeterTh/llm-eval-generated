cmake_minimum_required(VERSION 3.10)
project(cahn_hilliard_benchmark)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

enable_language(CUDA)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES 86)

find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)

add_executable(cahn_hilliard cahn_hilliard.cpp)
set_source_files_properties(cahn_hilliard.cpp PROPERTIES LANGUAGE CUDA)

# Host compiler flags via nvcc -Xcompiler
target_compile_options(cahn_hilliard PRIVATE
    -O3
    -Xcompiler=-march=native
    -Xcompiler=-fopenmp
    -Xcompiler=-Wall
    -Xcompiler=-Wextra
)

# Link MPI, OpenMP (gomp), and math
target_link_libraries(cahn_hilliard MPI::MPI_CXX gomp m)
