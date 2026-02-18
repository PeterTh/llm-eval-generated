cmake_minimum_required(VERSION 3.18)

if (NOT CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 70 CACHE STRING "CUDA architectures" FORCE)
endif()

project(cahn_hilliard_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)
find_package(CUDAToolkit REQUIRED)

add_executable(cahn_hilliard cahn_hilliard.cpp)
set_source_files_properties(cahn_hilliard.cpp PROPERTIES LANGUAGE CUDA)

# Enable optimizations
target_compile_options(cahn_hilliard PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-O3 -march=native -Wall -Wextra>
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 -Xcompiler=-fopenmp>
)

# Link math library and parallel runtimes
target_link_libraries(cahn_hilliard PRIVATE MPI::MPI_CXX OpenMP::OpenMP_CXX CUDA::cudart m)
