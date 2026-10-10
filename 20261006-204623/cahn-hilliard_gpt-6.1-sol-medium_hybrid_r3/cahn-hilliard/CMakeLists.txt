cmake_minimum_required(VERSION 3.18)
# Provide native code for common cluster accelerators and PTX for newer devices.
# A deployment can override this with -DCMAKE_CUDA_ARCHITECTURES=...
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 60 70 80 90)
endif()
project(cahn_hilliard_benchmark LANGUAGES CXX CUDA)

find_package(MPI REQUIRED COMPONENTS CXX)
find_package(OpenMP REQUIRED COMPONENTS CXX)
find_package(CUDAToolkit REQUIRED)

# Compile the existing source as CUDA without changing the executable name.
set_source_files_properties(cahn_hilliard.cpp PROPERTIES LANGUAGE CUDA)
add_executable(cahn_hilliard cahn_hilliard.cpp)
set_target_properties(cahn_hilliard PROPERTIES
    CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES
    CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED YES)
target_compile_options(cahn_hilliard PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3;--fmad=false;-Xcompiler=-Wall,-Wextra>
    $<$<COMPILE_LANGUAGE:CXX>:-O3;-Wall;-Wextra>)
# OpenMP's imported CXX target does not pass flags to a CUDA translation unit.
target_compile_options(cahn_hilliard PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=${OpenMP_CXX_FLAGS}>)
target_link_libraries(cahn_hilliard PRIVATE MPI::MPI_CXX OpenMP::OpenMP_CXX CUDA::cudart m)
