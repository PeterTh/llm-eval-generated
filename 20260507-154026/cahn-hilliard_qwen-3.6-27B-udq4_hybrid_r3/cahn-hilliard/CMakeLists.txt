cmake_minimum_required(VERSION 3.10)
project(cahn_hilliard_benchmark)

enable_language(CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)

set_source_files_properties(cahn_hilliard.cpp PROPERTIES LANGUAGE CUDA)

add_executable(cahn_hilliard cahn_hilliard.cpp)

# Enable optimizations
target_compile_options(cahn_hilliard PRIVATE -O3 -march=native -Wall -Wextra)
target_compile_options(cahn_hilliard PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-fopenmp>)

target_include_directories(cahn_hilliard PRIVATE ${MPI_CXX_INCLUDE_DIRS} ${OpenMP_CXX_INCLUDE_DIRS})
target_link_libraries(cahn_hilliard m MPI::MPI_CXX OpenMP::OpenMP_CXX cuda)
