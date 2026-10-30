cmake_minimum_required(VERSION 3.10)

project(qtclustering LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

add_executable(qtclustering qtclustering.cpp)

set_source_files_properties(qtclustering.cpp PROPERTIES LANGUAGE CUDA)

target_compile_options(qtclustering PRIVATE -O3 -march=native -Wall -Wextra)
target_compile_options(qtclustering PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3 -Xcompiler -fopenmp>)

find_package(MPI REQUIRED)
target_link_libraries(qtclustering MPI::MPI_CXX m cudart)
target_include_directories(qtclustering PRIVATE ${MPI_CXX_INCLUDE_DIRS})
target_link_options(qtclustering PRIVATE -fopenmp)
