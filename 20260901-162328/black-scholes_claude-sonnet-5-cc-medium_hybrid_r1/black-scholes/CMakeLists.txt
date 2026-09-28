cmake_minimum_required(VERSION 3.18)
project(black_scholes_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)

set(CMAKE_CUDA_ARCHITECTURES 86 CACHE STRING "CUDA architectures to target" FORCE)

add_executable(black_scholes black_scholes.cpp)

# Compile the translation unit as CUDA so that host (MPI/OpenMP) and device
# (GPU kernel) code can coexist in the single existing source file.
set_source_files_properties(black_scholes.cpp PROPERTIES LANGUAGE CUDA)

target_compile_options(black_scholes PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 -Xcompiler=-O3,-Wall,-Wextra,-fopenmp>
)

target_link_libraries(black_scholes PRIVATE MPI::MPI_CXX OpenMP::OpenMP_CXX m)
