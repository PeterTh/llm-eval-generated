cmake_minimum_required(VERSION 3.18)
project(floydwarshall_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# The benchmark is tuned for the common Ampere data-center GPU target used by
# this workload. Users can override this cache variable for another GPU.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 86 CACHE STRING "CUDA architectures")
endif()

add_executable(floydwarshall floydwarshall.cpp)

# Keep the existing source file and executable name while compiling the
# implementation with nvcc.
set_source_files_properties(floydwarshall.cpp PROPERTIES LANGUAGE CUDA)

# Enable high optimization for the CUDA kernels and the host-side benchmark.
target_compile_options(floydwarshall PRIVATE
    "$<$<COMPILE_LANGUAGE:CUDA>:-O3;--use_fast_math>"
    "$<$<COMPILE_LANGUAGE:CXX>:-O3;-march=native;-Wall;-Wextra>"
)

# Link math library
target_link_libraries(floydwarshall PRIVATE m)
