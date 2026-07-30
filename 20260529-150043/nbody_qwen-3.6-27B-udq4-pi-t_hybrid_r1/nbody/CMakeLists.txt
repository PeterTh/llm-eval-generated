cmake_minimum_required(VERSION 3.10)
project(nbody_benchmark)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Enable CUDA support
enable_language(CUDA)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES 86)  # RTX 3090 (Ampere)

# Find MPI
find_package(MPI REQUIRED)

add_executable(nbody nbody.cpp)

# Compile nbody.cpp with nvcc (CUDA) so kernels are processed correctly
set_source_files_properties(nbody.cpp PROPERTIES LANGUAGE CUDA)

# Compile with MPI, OpenMP, and CUDA optimizations
target_compile_options(nbody PRIVATE -O3 -march=native -Wall -Wextra -Xcompiler=-fopenmp)
target_compile_options(nbody PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>")

# Link MPI, CUDA runtime, and math
# OpenMP is handled by -fopenmp in compile options (GCC links libgomp implicitly)
target_link_libraries(nbody MPI::MPI_CXX MPI::MPI_C cudart m gomp)

# Set CUDA compiler flags for performance
set_target_properties(nbody PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)
