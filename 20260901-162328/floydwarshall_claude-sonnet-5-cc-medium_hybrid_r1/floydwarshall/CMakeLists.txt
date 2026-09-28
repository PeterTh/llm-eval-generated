cmake_minimum_required(VERSION 3.18)
project(floydwarshall_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT DEFINED CMAKE_CUDA_STANDARD)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 70 75 80 86 90)
endif()

find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)
find_package(CUDAToolkit REQUIRED)

add_executable(floydwarshall floydwarshall.cpp)

# Compile floydwarshall.cpp with nvcc so it can contain CUDA kernels directly.
set_source_files_properties(floydwarshall.cpp PROPERTIES LANGUAGE CUDA)

target_include_directories(floydwarshall PRIVATE ${MPI_CXX_INCLUDE_DIRS})

# Enable optimizations. OpenMP and other host-only flags are forwarded to the
# host compiler via -Xcompiler since the translation unit is built by nvcc.
target_compile_options(floydwarshall PRIVATE
    -O3
    -Xcompiler=-O3,-Wall,-Wextra,-march=native,-fopenmp
)

target_link_options(floydwarshall PRIVATE $<DEVICE_LINK:-Xcompiler=-fopenmp> $<HOST_LINK:-fopenmp>)

# Link MPI, CUDA runtime, and math library
target_link_libraries(floydwarshall PRIVATE ${MPI_CXX_LIBRARIES} CUDA::cudart m)
