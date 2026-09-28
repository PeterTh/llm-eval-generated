cmake_minimum_required(VERSION 3.24)
project(black_scholes_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES native)
endif()

find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)

add_executable(black_scholes black_scholes.cpp)

# The whole source file is compiled by nvcc so that __host__ __device__
# functions and the CUDA kernel can share the same translation unit.
set_source_files_properties(black_scholes.cpp PROPERTIES LANGUAGE CUDA)

target_include_directories(black_scholes PRIVATE ${MPI_CXX_INCLUDE_DIRS})

# Enable optimizations; forward host-side flags (OpenMP, native arch) through nvcc
target_compile_options(black_scholes PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 -Xcompiler=-O3 -Xcompiler=-march=native -Xcompiler=-fopenmp --extended-lambda>
)
# Link math library, MPI, and OpenMP runtime
target_link_libraries(black_scholes PRIVATE MPI::MPI_CXX m gomp)
