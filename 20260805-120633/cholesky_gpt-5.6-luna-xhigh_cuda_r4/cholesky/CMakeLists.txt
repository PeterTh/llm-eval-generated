cmake_minimum_required(VERSION 3.18)
project(cholesky_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# The benchmark is run on Ampere GPUs in the target environment.  Keep this
# as a cache-overridable default so the same source can be rebuilt for another
# CUDA device without changing the benchmark's interface.  CMake may seed
# CMAKE_CUDA_ARCHITECTURES with an old compiler default before this file is
# evaluated, so use a benchmark-specific cache variable for the override.
set(CHOLESKY_CUDA_ARCHITECTURES 86 CACHE STRING
    "CUDA architectures used to build the Cholesky benchmark")
set(CMAKE_CUDA_ARCHITECTURES "${CHOLESKY_CUDA_ARCHITECTURES}")

# Keep the original .cpp filename and executable name while compiling the
# implementation unconditionally as CUDA code.
set_source_files_properties(cholesky.cpp PROPERTIES LANGUAGE CUDA)

add_executable(cholesky cholesky.cpp)

# Enable optimizations
target_compile_options(cholesky PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3>
    $<$<COMPILE_LANGUAGE:CXX>:-O3;-march=native;-Wall;-Wextra>
)

# Link math library
target_link_libraries(cholesky m)
