cmake_minimum_required(VERSION 3.18)
project(black_scholes_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)
find_package(CUDAToolkit REQUIRED)

add_executable(black_scholes black_scholes.cpp)

# Compile the translation unit with nvcc so it can host both the CUDA
# kernel and the MPI/OpenMP driver code in a single existing source file.
set_source_files_properties(black_scholes.cpp PROPERTIES LANGUAGE CUDA)

set_target_properties(black_scholes PROPERTIES
    CUDA_ARCHITECTURES "70;75;80;86;89;90"
    CUDA_SEPARABLE_COMPILATION OFF
)

target_include_directories(black_scholes PRIVATE ${MPI_CXX_INCLUDE_DIRS})

# Enable optimizations
target_compile_options(black_scholes PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 --use_fast_math -Xcompiler=-fopenmp,-O3,-march=native,-Wall,-Wextra>
)

target_link_libraries(black_scholes PRIVATE
    MPI::MPI_CXX
    OpenMP::OpenMP_CXX
    CUDA::cudart
    m
)
