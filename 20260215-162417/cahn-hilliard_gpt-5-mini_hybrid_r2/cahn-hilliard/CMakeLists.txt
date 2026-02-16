cmake_minimum_required(VERSION 3.18)
project(cahn_hilliard_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)

add_executable(cahn_hilliard cahn_hilliard.cpp)
# Force compilation of the single source file as CUDA so kernels can live in it
set_source_files_properties(cahn_hilliard.cpp PROPERTIES LANGUAGE CUDA)

# Compiler options for C++ and CUDA
target_compile_options(cahn_hilliard PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-O3 -march=native -Wall -Wextra>
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 -use_fast_math -lineinfo>
)

# Link math, MPI, OpenMP and CUDA runtime
target_link_libraries(cahn_hilliard PRIVATE m MPI::MPI_CXX OpenMP::OpenMP_CXX -lcudart)
