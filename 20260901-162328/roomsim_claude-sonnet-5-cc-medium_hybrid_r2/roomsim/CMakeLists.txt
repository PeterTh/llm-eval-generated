cmake_minimum_required(VERSION 3.18)
project(roomsim_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 70 75 80 86 90)
endif()

find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)

add_executable(roomsim roomsim.cpp)

# The single source file is compiled by nvcc as CUDA so that host code
# (MPI/OpenMP driven) and device kernels can coexist in one translation unit.
set_source_files_properties(roomsim.cpp PROPERTIES LANGUAGE CUDA)

target_include_directories(roomsim PRIVATE ${MPI_CXX_INCLUDE_DIRS})

# MPI and OpenMP compile/link flags are passed through -Xcompiler since nvcc
# does not understand raw host-compiler flags like -fopenmp directly.
target_compile_options(roomsim PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3;--expt-relaxed-constexpr;--extended-lambda;-Xcompiler=-O3,-march=native,-Wall,-Wextra,-fopenmp>
)

target_link_options(roomsim PRIVATE -fopenmp)

target_link_libraries(roomsim PRIVATE ${MPI_CXX_LIBRARIES} gomp m cudart)
