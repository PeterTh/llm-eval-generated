cmake_minimum_required(VERSION 3.10)

project(qtclustering)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Enable CUDA support
enable_language(CUDA)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES 86)

# Find MPI
find_package(MPI REQUIRED)

add_executable(qtclustering qtclustering.cpp)

# Set CUDA properties for the source
set_source_files_properties(qtclustering.cpp PROPERTIES LANGUAGE CUDA)

# Compile options: O3 optimization, native arch, OpenMP, CUDA flags
target_compile_options(qtclustering PRIVATE
    -O3
    -march=native
    -Wall
    -Wextra
    -fopenmp
)

# Set CUDA-specific compile flags
set_target_properties(qtclustering PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)
target_compile_options(qtclustering PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
)

# Link: MPI, OpenMP, CUDA runtime, math
target_link_libraries(qtclustering
    MPI::MPI_CXX
    gomp
    cudart
    m
)

# Include directories
target_include_directories(qtclustering PRIVATE ${MPI_CXX_INCLUDE_PATH})
