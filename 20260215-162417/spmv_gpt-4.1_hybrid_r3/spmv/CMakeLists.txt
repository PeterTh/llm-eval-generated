cmake_minimum_required(VERSION 3.10)
project(spmv_benchmark LANGUAGES CXX CUDA)


set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(MPI REQUIRED)
find_package(CUDA REQUIRED)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Compile CUDA code as part of the C++ file
set_source_files_properties(spmv.cpp PROPERTIES LANGUAGE CUDA)

add_executable(spmv spmv.cpp)

if (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(spmv PRIVATE -O3 -march=native -Wall -Wextra -fopenmp)
endif()

if (CUDA_FOUND)
    target_link_libraries(spmv m MPI::MPI_CXX ${CUDA_CUDART_LIBRARY})
    target_include_directories(spmv PRIVATE ${CUDA_INCLUDE_DIRS})
endif()
