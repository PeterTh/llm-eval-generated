cmake_minimum_required(VERSION 3.24)

# Include native code for common cluster GPUs and PTX for newer devices.
# A cluster build can override this with -DCMAKE_CUDA_ARCHITECTURES=...
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 70-real 80-real 86)
endif()
project(black_scholes_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(MPI REQUIRED COMPONENTS CXX)
find_package(OpenMP REQUIRED COMPONENTS CXX)
find_package(CUDAToolkit REQUIRED)

# Keep the original filename while compiling the kernels with nvcc.
set_source_files_properties(black_scholes.cpp PROPERTIES LANGUAGE CUDA)
add_executable(black_scholes black_scholes.cpp)

# Preserve double precision and the formula's arithmetic ordering.
target_compile_options(black_scholes PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3;--fmad=false;-Xcompiler=-fopenmp,-Wall,-Wextra>
)
target_compile_definitions(black_scholes PRIVATE OMPI_SKIP_MPICXX MPICH_SKIP_MPICXX)

target_link_libraries(black_scholes PRIVATE
    MPI::MPI_CXX OpenMP::OpenMP_CXX CUDA::cudart m
)
