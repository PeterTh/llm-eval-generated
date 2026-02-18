cmake_minimum_required(VERSION 3.10)
if (NOT DEFINED CMAKE_CUDA_ARCHITECTURES OR CMAKE_CUDA_ARCHITECTURES STREQUAL "")
    set(CMAKE_CUDA_ARCHITECTURES 70 CACHE STRING "CUDA architectures" FORCE)
endif()
project(nbody_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)

add_executable(nbody nbody.cpp)
set_source_files_properties(nbody.cpp PROPERTIES LANGUAGE CUDA)
set_target_properties(nbody PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# Enable optimizations
# CUDA compilation happens via NVCC; pass host flags explicitly.
target_compile_options(nbody PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-O3 -march=native -Wall -Wextra>
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 -lineinfo -Xcompiler=-march=native -Xcompiler=-Wall -Xcompiler=-Wextra -Xcompiler=-fopenmp>
)

# Link math library and parallel runtimes
target_link_libraries(nbody PRIVATE MPI::MPI_CXX OpenMP::OpenMP_CXX m)
