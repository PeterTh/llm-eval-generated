cmake_minimum_required(VERSION 3.10)

project(qtclustering LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(qtclustering qtclustering.cpp)
find_package(MPI REQUIRED COMPONENTS CXX)
find_package(OpenMP REQUIRED COMPONENTS CXX)
set_source_files_properties(qtclustering.cpp PROPERTIES LANGUAGE CUDA)
target_link_libraries(qtclustering m MPI::MPI_CXX OpenMP::OpenMP_CXX)
target_compile_options(qtclustering PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3;--use_fast_math>)

target_compile_options(qtclustering PRIVATE -O3 -march=native -Wall -Wextra)

# Link math library
