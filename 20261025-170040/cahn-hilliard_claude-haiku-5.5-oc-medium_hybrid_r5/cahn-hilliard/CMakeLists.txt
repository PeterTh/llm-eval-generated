cmake_minimum_required(VERSION 3.24)
project(cahn_hilliard_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Target GPU architecture(s); override with -DCMAKE_CUDA_ARCHITECTURES=<list>
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES native)
endif()

# Hybrid parallelization: MPI (inter-node), OpenMP (one thread per GPU), CUDA (kernels)
find_package(MPI REQUIRED COMPONENTS CXX)
find_package(OpenMP REQUIRED COMPONENTS CXX)
find_package(CUDAToolkit REQUIRED)

# The solver is compiled as CUDA source (nvcc drives the host compiler for the
# MPI/OpenMP host code), so the executable name and source file are unchanged.
add_executable(cahn_hilliard cahn_hilliard.cpp)
set_source_files_properties(cahn_hilliard.cpp PROPERTIES LANGUAGE CUDA)

# Optimizations and warnings for host code (forwarded through nvcc) and device code
target_compile_options(cahn_hilliard PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler;-O3>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler;-march=native>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler;-Wall>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler;-Wextra>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler;${OpenMP_CXX_FLAGS}>
    # No implicit contraction on the device: all fused operations are explicit fma() calls
    $<$<COMPILE_LANGUAGE:CUDA>:--fmad=false>
)

target_link_options(cahn_hilliard PRIVATE ${OpenMP_CXX_FLAGS})

# Link MPI, CUDA runtime and math library
target_link_libraries(cahn_hilliard PRIVATE MPI::MPI_CXX CUDA::cudart m)
