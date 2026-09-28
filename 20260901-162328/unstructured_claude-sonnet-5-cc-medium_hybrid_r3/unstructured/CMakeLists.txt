cmake_minimum_required(VERSION 3.18)
project(unstructured_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 70;75;80;86)
endif()

find_package(MPI REQUIRED COMPONENTS CXX)

add_executable(unstructured unstructured.cpp)

# Compile the single source file as CUDA so it can host both the MPI/OpenMP
# driver code and the __global__ compute kernel without introducing new files.
set_source_files_properties(unstructured.cpp PROPERTIES LANGUAGE CUDA)

target_include_directories(unstructured PRIVATE ${MPI_CXX_INCLUDE_DIRS})

# Enable optimizations; forward host-compiler flags (warnings, OpenMP, native tuning) via -Xcompiler
target_compile_options(unstructured PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 --use_fast_math -Xcompiler=-O3,-Wall,-Wextra,-fopenmp,-march=native>
)

target_link_options(unstructured PRIVATE -fopenmp)

# Link math, MPI and CUDA runtime libraries
target_link_libraries(unstructured PRIVATE m gomp ${MPI_CXX_LIBRARIES} cudart)
