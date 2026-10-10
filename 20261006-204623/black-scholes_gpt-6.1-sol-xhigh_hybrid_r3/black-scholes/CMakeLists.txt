cmake_minimum_required(VERSION 3.20)

# Include native code for common cluster GPUs and PTX for newer devices.
# A cluster build can override this with -DCMAKE_CUDA_ARCHITECTURES=... .
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES AND NOT DEFINED ENV{CUDAARCHS})
    set(CMAKE_CUDA_ARCHITECTURES 70 80 86)
endif()

project(black_scholes_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(MPI REQUIRED COMPONENTS CXX)
find_package(OpenMP REQUIRED COMPONENTS CXX)
find_package(CUDAToolkit REQUIRED)

# Compile the existing file as CUDA; no additional source file is needed.
set_source_files_properties(black_scholes.cpp PROPERTIES LANGUAGE CUDA)
add_executable(black_scholes black_scholes.cpp)

target_compile_definitions(black_scholes PRIVATE OMPI_SKIP_MPICXX=1 MPICH_SKIP_MPICXX=1)
target_compile_options(black_scholes PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3;--fmad=false;-Xcompiler=-O3,-ffp-contract=off,-Wall,-Wextra>
)

# OpenMP's imported target supplies the host runtime, but its CXX compile
# options must also be forwarded through nvcc for this CUDA translation unit.
separate_arguments(BLACK_SCHOLES_OPENMP_FLAGS NATIVE_COMMAND "${OpenMP_CXX_FLAGS}")
foreach(flag IN LISTS BLACK_SCHOLES_OPENMP_FLAGS)
    target_compile_options(black_scholes PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=${flag}>)
endforeach()

target_link_libraries(black_scholes PRIVATE MPI::MPI_CXX OpenMP::OpenMP_CXX CUDA::cudart m)
