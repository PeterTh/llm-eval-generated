cmake_minimum_required(VERSION 3.18)

project(qtclustering LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(CUDAToolkit REQUIRED)

add_executable(qtclustering qtclustering.cpp)

# Ensure the single source is compiled with CUDA (nvcc) so device kernels in the .cpp are built
set_source_files_properties(qtclustering.cpp PROPERTIES LANGUAGE CUDA)

# Compiler options for host and device
target_compile_options(qtclustering PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-O3 -march=native -Wall -Wextra>
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 --use_fast_math>
)

# Link CUDA runtime and math library
target_link_libraries(qtclustering PRIVATE CUDAToolkit::cudart m)
