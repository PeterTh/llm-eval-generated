cmake_minimum_required(VERSION 3.24)

# Target the GPUs present on the build machine unless overridden with -DCMAKE_CUDA_ARCHITECTURES
# (must be set before project() so that CMake does not fill in its compiler default first)
if(NOT CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES native)
endif()

project(nbody_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(MPI REQUIRED COMPONENTS CXX)
find_package(OpenMP REQUIRED COMPONENTS CXX)
find_package(CUDAToolkit REQUIRED)

# The single source is compiled as CUDA so that the kernels live alongside the host code
add_executable(nbody nbody.cpp)
set_source_files_properties(nbody.cpp PROPERTIES LANGUAGE CUDA)

# Optimization and warning flags (host flags forwarded through nvcc).
# -fmad=false: device multiply-adds are only emitted where the kernels request them
#   explicitly, so GPU arithmetic matches the reference build bit-for-bit.
# -ffp-contract=fast: host arithmetic uses the same FMA contraction as the reference build.
target_compile_options(nbody PRIVATE
    -O3
    -fmad=false
    -Xcompiler=-march=native
    -Xcompiler=-ffp-contract=fast
    -Xcompiler=-Wall
    -Xcompiler=-Wextra
    -Xcompiler=-fopenmp
)

target_link_libraries(nbody PRIVATE
    MPI::MPI_CXX
    OpenMP::OpenMP_CXX
    CUDA::cudart
    m
)
