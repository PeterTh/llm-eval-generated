cmake_minimum_required(VERSION 3.18)
project(floydwarshall_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# The source remains .cpp to preserve the benchmark layout; compile it with
# nvcc so the executable contains the CUDA kernel and the MPI/OpenMP host code.
set_source_files_properties(floydwarshall.cpp PROPERTIES LANGUAGE CUDA)

find_package(MPI REQUIRED COMPONENTS CXX)
find_package(OpenMP REQUIRED COMPONENTS CXX)
find_package(CUDAToolkit REQUIRED)

add_executable(floydwarshall floydwarshall.cpp)

# Enable optimized host/device code.  CUDA architectures can be overridden at
# configure time with -DCMAKE_CUDA_ARCHITECTURES=... for a specific cluster.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 75 80 86 89 CACHE STRING
        "CUDA architectures generated for the Floyd-Warshall executable")
endif()
set_target_properties(floydwarshall PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)
target_compile_options(floydwarshall PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-O3;-march=native;-Wall;-Wextra>
    $<$<COMPILE_LANGUAGE:CUDA>:-O3;-Xcompiler=-O3;-Xcompiler=-fopenmp;-Xcompiler=-Wall;-Xcompiler=-Wextra>
)
target_compile_definitions(floydwarshall PRIVATE OMPI_SKIP_MPICXX=1)

target_link_libraries(floydwarshall PRIVATE
    MPI::MPI_CXX
    OpenMP::OpenMP_CXX
    CUDA::cudart
    m
)
