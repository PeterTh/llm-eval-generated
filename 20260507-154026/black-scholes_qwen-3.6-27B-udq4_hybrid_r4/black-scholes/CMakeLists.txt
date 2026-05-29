cmake_minimum_required(VERSION 3.10)
project(black_scholes_benchmark)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find required parallelization libraries
find_package(MPI REQUIRED)
find_package(CUDA REQUIRED)

# Determine OpenMP
include(FindOpenMP)
if(NOT OPENMP_FOUND)
    # Fallback: manually search for libomp
    find_library(OPENMP_LIB NAMES omp libomp PATHS /usr/lib /usr/local/lib /opt/homebrew/lib)
    if(OPENMP_LIB)
        set(OPENMP_CXX_FLAGS "-fopenmp")
        set(OPENMP_CXX_LIB_NAMES "${OPENMP_LIB}")
        set(OPENMP_FOUND TRUE)
    endif()
endif()

# Enable CUDA
enable_language(CUDA)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Generate CUDA device code from the .cpp file (nvcc handles __device__ functions)
add_executable(black_scholes black_scholes.cu)

# Compile flags: O3, native arch, OpenMP, MPI
target_compile_options(black_scholes PRIVATE
    -O3 -march=native -Wall -Wextra
    $<$<COMPILE_LANGUAGE:CXX>:${OpenMP_CXX_FLAGS}>
)

# CUDA compile flags
set_target_properties(black_scholes PROPERTIES
    CUDA_ARCHITECTURES "86"
)
target_compile_options(black_scholes PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda -Xcompiler=-O3>
)

# Link: MPI, OpenMP, CUDA runtime, math
target_link_libraries(black_scholes
    MPI::MPI_CXX
    m
    $<$<BOOL:${OpenMP_CXX_LIB_NAMES}>:${OpenMP_CXX_LIB_NAMES}>
)
