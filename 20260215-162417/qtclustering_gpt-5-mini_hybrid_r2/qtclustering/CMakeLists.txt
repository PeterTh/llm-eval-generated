cmake_minimum_required(VERSION 3.18)

project(qtclustering LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)
add_executable(qtclustering qtclustering.cpp)

# Default to no CUDA in this environment; CUDA will be enabled on systems with a CUDA-capable toolchain by adjusting CMake.
# If building on a CUDA-capable cluster, set -DHAS_CUDA=1 or enable CUDAToolkit in the top-level CMake to compile kernels.
target_compile_definitions(qtclustering PRIVATE HAS_CUDA=0)

# Compiler flags for C++ and CUDA
target_compile_options(qtclustering PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-O3 -march=native -Wall -Wextra>
    $<$<COMPILE_LANGUAGE:CUDA>:-O3>
)

# Link MPI and OpenMP and math
target_link_libraries(qtclustering PRIVATE MPI::MPI_CXX OpenMP::OpenMP_CXX m)

target_include_directories(qtclustering PRIVATE ${MPI_CXX_INCLUDE_DIRS})

# Define a macro to indicate hybrid build (CPU fallback possible)
target_compile_definitions(qtclustering PRIVATE HYBRID_MPI_OMP_CUDA=1)
