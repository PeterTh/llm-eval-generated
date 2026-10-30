cmake_minimum_required(VERSION 3.10)
project(floydwarshall_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 14)

find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)

add_executable(floydwarshall floydwarshall.cpp)
set_source_files_properties(floydwarshall.cpp PROPERTIES LANGUAGE CUDA)
set_target_properties(floydwarshall PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

target_compile_options(floydwarshall PRIVATE -O3 -march=native -Wall -Wextra)
target_link_libraries(floydwarshall PRIVATE MPI::MPI_CXX OpenMP::OpenMP_CXX m)
target_include_directories(floydwarshall PRIVATE ${MPI_CXX_INCLUDE_DIRS})
