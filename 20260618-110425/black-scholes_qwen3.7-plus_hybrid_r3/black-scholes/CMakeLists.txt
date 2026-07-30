cmake_minimum_required(VERSION 3.18)
project(black_scholes_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Find MPI
find_package(MPI REQUIRED)

# Find OpenMP
find_package(OpenMP REQUIRED)

add_executable(black_scholes black_scholes.cpp)

# Treat .cpp file as CUDA source so nvcc compiles it
set_source_files_properties(black_scholes.cpp PROPERTIES LANGUAGE CUDA)

# CUDA compilation flags
target_compile_options(black_scholes PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 --use_fast_math -Xcompiler -march=native,-Wall,-Wextra>
)

# Pass MPI and OpenMP flags through to the host compiler via nvcc
target_compile_options(black_scholes PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:SHELL:${OpenMP_CXX_FLAGS}>
)

# Include directories for MPI (needed by nvcc)
target_include_directories(black_scholes PRIVATE ${MPI_CXX_INCLUDE_DIRS})

# Link libraries
target_link_libraries(black_scholes 
    m 
    MPI::MPI_CXX 
    OpenMP::OpenMP_CXX
)

# Pass OpenMP link flag
target_link_options(black_scholes PRIVATE ${OpenMP_CXX_FLAGS})
