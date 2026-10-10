cmake_minimum_required(VERSION 3.18)
# Respect explicit toolchain choices. CUDA 12.0 supports GCC through version 12,
# so prefer that host compiler when it is available on newer Linux systems.
if(NOT DEFINED CMAKE_CUDA_HOST_COMPILER AND NOT DEFINED ENV{CUDAHOSTCXX})
    find_program(NBODY_CUDA_HOST_COMPILER NAMES g++-12)
    if(NBODY_CUDA_HOST_COMPILER)
        set(CMAKE_CUDA_HOST_COMPILER "${NBODY_CUDA_HOST_COMPILER}")
    endif()
endif()
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES AND NOT DEFINED ENV{CUDAARCHS}
   AND CMAKE_VERSION VERSION_GREATER_EQUAL 3.24)
    set(CMAKE_CUDA_ARCHITECTURES native)
endif()
project(nbody_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Compile the existing source with nvcc while preserving the executable name.
set_source_files_properties(nbody.cpp PROPERTIES LANGUAGE CUDA)
add_executable(nbody nbody.cpp)

# Preserve double-precision arithmetic and accumulation order: CUDA's default
# fused multiply-add changes the original simulation's rounding behavior.
target_compile_options(nbody PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3;--fmad=false;-Xcompiler=-Wall,-Wextra>)
target_link_libraries(nbody PRIVATE m)
