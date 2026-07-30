cmake_minimum_required(VERSION 3.18)
project(cahn_hilliard_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)

set(CMAKE_CUDA_ARCHITECTURES "70;75;80;86;90")

add_executable(cahn_hilliard cahn_hilliard.cpp)
set_source_files_properties(cahn_hilliard.cpp PROPERTIES LANGUAGE CUDA)

target_compile_options(cahn_hilliard PRIVATE
    -O3
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-march=native;-Xcompiler=-Wall;-Xcompiler=-Wextra;-Xcompiler=-fopenmp>
    $<$<COMPILE_LANGUAGE:CXX>:-march=native;-Wall;-Wextra;-fopenmp>
)

target_link_libraries(cahn_hilliard
    PRIVATE
    MPI::MPI_CXX
    OpenMP::OpenMP_CXX
    m
)
