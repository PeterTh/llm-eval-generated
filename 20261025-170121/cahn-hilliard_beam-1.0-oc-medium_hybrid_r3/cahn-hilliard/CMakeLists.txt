cmake_minimum_required(VERSION 3.10)
project(cahn_hilliard_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)

find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)

add_executable(cahn_hilliard cahn_hilliard.cpp)
set_source_files_properties(cahn_hilliard.cpp PROPERTIES LANGUAGE CUDA)

target_include_directories(cahn_hilliard PRIVATE ${MPI_CXX_INCLUDE_DIRS})
target_compile_options(cahn_hilliard PRIVATE -O3 -march=native -Wall -Wextra)
target_link_libraries(cahn_hilliard PRIVATE MPI::MPI_CXX OpenMP::OpenMP_CXX m cuda)
