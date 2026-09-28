cmake_minimum_required(VERSION 3.18)
project(black_scholes_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 70 75 80 86 89)
endif()

find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)

add_executable(black_scholes black_scholes.cpp)

# Compile the source as CUDA so it can host both device kernels and the
# MPI/OpenMP-parallel host code in a single translation unit.
set_source_files_properties(black_scholes.cpp PROPERTIES LANGUAGE CUDA)
set_target_properties(black_scholes PROPERTIES
    LINKER_LANGUAGE CUDA
    CUDA_SEPARABLE_COMPILATION OFF
)

target_include_directories(black_scholes PRIVATE ${MPI_CXX_INCLUDE_DIRS})

# Enable optimizations. Host-only flags must be routed through -Xcompiler
# since this target is compiled/linked with nvcc.
target_compile_options(black_scholes PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 -Xcompiler=-O3,-march=native,-fopenmp,-Wall,-Wextra>
)
target_link_options(black_scholes PRIVATE -fopenmp)

target_link_libraries(black_scholes PRIVATE ${MPI_CXX_LIBRARIES} m)
