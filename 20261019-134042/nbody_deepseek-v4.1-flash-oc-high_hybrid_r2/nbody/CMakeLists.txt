cmake_minimum_required(VERSION 3.24)
project(nbody_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(MPI REQUIRED COMPONENTS CXX)
find_package(OpenMP REQUIRED COMPONENTS CXX)
find_package(CUDAToolkit REQUIRED)

# Target the common datacenter/workstation architectures (PTX for the newest
# one keeps the binary forward compatible with newer GPUs).
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES "70;75;80;86;89")
endif()

add_executable(nbody nbody.cpp)

# The force kernel lives in the same translation unit, so compile it as CUDA.
set_source_files_properties(nbody.cpp PROPERTIES LANGUAGE CUDA)
set_target_properties(nbody PROPERTIES CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}")

# Enable optimizations.  Host code must use the same floating point
# contraction as the reference build; device code uses explicit fma() calls
# and disables implicit contraction so both paths agree numerically.
target_compile_options(nbody PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3>
    $<$<COMPILE_LANGUAGE:CUDA>:--fmad=false>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-O3>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-march=native>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wextra>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-fopenmp>
)

target_include_directories(nbody PRIVATE ${MPI_CXX_INCLUDE_DIRS})

# Link MPI, OpenMP (libgomp), CUDA runtime and the math library.
target_link_libraries(nbody PRIVATE
    MPI::MPI_CXX
    OpenMP::OpenMP_CXX
    CUDA::cudart
    m
)
