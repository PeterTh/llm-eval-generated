cmake_minimum_required(VERSION 3.18)
project(black_scholes_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES
   OR "${CMAKE_CUDA_ARCHITECTURES}" STREQUAL ""
   OR "${CMAKE_CUDA_ARCHITECTURES}" STREQUAL "52")
    set(CMAKE_CUDA_ARCHITECTURES "70;75;80;86")
endif()

find_package(MPI REQUIRED COMPONENTS CXX)
find_package(OpenMP REQUIRED COMPONENTS CXX)

# The benchmark is a single CUDA translation unit; the host side uses OpenMP
# and MPI for node/rank decomposition.
set_source_files_properties(black_scholes.cpp PROPERTIES LANGUAGE CUDA)

add_executable(black_scholes black_scholes.cpp)

target_compile_options(black_scholes PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3;-Xcompiler=-march=native;-Xcompiler=-fopenmp>
    $<$<COMPILE_LANGUAGE:CXX>:-O3;-march=native;-fopenmp>
)

target_link_libraries(black_scholes PRIVATE MPI::MPI_CXX ${OpenMP_CXX_LIBRARIES} m)
