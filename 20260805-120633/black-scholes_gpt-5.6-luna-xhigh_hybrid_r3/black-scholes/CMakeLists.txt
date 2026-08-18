cmake_minimum_required(VERSION 3.24)
project(black_scholes_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Compile for the local accelerator by default.  Cross-compiling users can
# override this cache variable with one or more explicit architectures.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES OR CMAKE_CUDA_ARCHITECTURES STREQUAL "")
    set(CMAKE_CUDA_ARCHITECTURES native)
endif()

find_package(MPI REQUIRED COMPONENTS CXX)
find_package(OpenMP REQUIRED)

# Keep the existing source and executable name while compiling the translation
# unit with nvcc so it can contain both the MPI/OpenMP host code and CUDA
# kernels.  Users may set CMAKE_CUDA_ARCHITECTURES for the target cluster.
set_source_files_properties(black_scholes.cpp PROPERTIES LANGUAGE CUDA)

add_executable(black_scholes black_scholes.cpp)

# Enable aggressive host/device optimization.  OpenMP is passed explicitly to
# nvcc's host compiler because the source is intentionally a CUDA translation
# unit with a .cpp extension.
target_compile_options(black_scholes PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-O3;-march=native;-Wall;-Wextra>
    $<$<COMPILE_LANGUAGE:CUDA>:-O3;--use_fast_math;-Xcompiler=-fopenmp;-Xcompiler=-Wall;-Xcompiler=-Wextra>
)

target_link_libraries(black_scholes PRIVATE
    MPI::MPI_CXX
    OpenMP::OpenMP_CXX
    m
)
