cmake_minimum_required(VERSION 3.10)
project(cholesky_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)

add_executable(cholesky cholesky.cpp)
set_source_files_properties(cholesky.cpp PROPERTIES LANGUAGE CUDA)

target_compile_options(cholesky PRIVATE -O3 -march=native -Wall -Wextra -Xcompiler=-fopenmp)
target_link_libraries(cholesky PRIVATE MPI::MPI_CXX OpenMP::OpenMP_CXX cudart m)
