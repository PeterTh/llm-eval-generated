cmake_minimum_required(VERSION 3.18)
project(nbody_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 86)
endif()

# Compile the (otherwise plain C++) source as CUDA so device kernels, MPI and
# OpenMP host code can all live in the single existing source file.
set_source_files_properties(nbody.cpp PROPERTIES LANGUAGE CUDA)

add_executable(nbody nbody.cpp)

set_target_properties(nbody PROPERTIES
    CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
    CUDA_SEPARABLE_COMPILATION OFF
)

target_include_directories(nbody PRIVATE ${MPI_CXX_INCLUDE_DIRS})

# Enable optimizations for both the host (via -Xcompiler) and device code.
target_compile_options(nbody PRIVATE
    -O3
    -Xcompiler=-O3,-march=native,-Wall,-Wextra,-fopenmp
)

target_link_libraries(nbody PRIVATE ${MPI_CXX_LIBRARIES} m)
target_link_options(nbody PRIVATE -fopenmp)
