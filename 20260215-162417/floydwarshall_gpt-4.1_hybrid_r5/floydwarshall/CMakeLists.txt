cmake_minimum_required(VERSION 3.10)
project(floydwarshall_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)
enable_language(CUDA)

add_executable(floydwarshall floydwarshall.cpp)

# Enable optimizations
target_compile_options(floydwarshall PRIVATE -O3 -march=native -Wall -Wextra)

# Link math library and parallel libraries
target_link_libraries(floydwarshall m MPI::MPI_CXX OpenMP::OpenMP_CXX cuda)

target_compile_definitions(floydwarshall PRIVATE USE_MPI USE_OPENMP USE_CUDA)

target_compile_options(floydwarshall PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-arch=sm_60>)
