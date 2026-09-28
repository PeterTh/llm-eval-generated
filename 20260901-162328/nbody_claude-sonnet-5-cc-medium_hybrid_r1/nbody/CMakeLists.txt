cmake_minimum_required(VERSION 3.18)
project(nbody_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(NOT CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 70 75 80 86 90)
endif()

find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)

# Compile the single source file as CUDA so it can mix host (MPI/OpenMP) and
# device (kernel) code in one translation unit.
set_source_files_properties(nbody.cpp PROPERTIES LANGUAGE CUDA)

add_executable(nbody nbody.cpp)

set_target_properties(nbody PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# Enable optimizations
target_compile_options(nbody PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 -Xcompiler=-O3,-Wall,-Wextra,-march=native,-fopenmp>
)

target_include_directories(nbody PRIVATE ${MPI_CXX_INCLUDE_DIRS})
target_link_libraries(nbody PRIVATE m MPI::MPI_CXX OpenMP::OpenMP_CXX)
