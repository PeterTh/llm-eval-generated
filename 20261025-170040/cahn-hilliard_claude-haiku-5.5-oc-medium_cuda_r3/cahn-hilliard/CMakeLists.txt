cmake_minimum_required(VERSION 3.18)
project(cahn_hilliard_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Target the GPU(s) present on the build machine unless the user overrides it
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES native)
endif()

find_package(CUDAToolkit REQUIRED)

add_executable(cahn_hilliard cahn_hilliard.cpp)

# The solver is GPU-only: compile the source with the CUDA compiler
set_source_files_properties(cahn_hilliard.cpp PROPERTIES LANGUAGE CUDA)

# --fmad=false keeps a*b+c unfused so the GPU results are bit-reproducible
# with a strict (non-FMA) IEEE evaluation of the same expressions.
target_compile_options(cahn_hilliard PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 --fmad=false -Xcompiler=-Wall,-Wextra>
)

target_link_libraries(cahn_hilliard PRIVATE CUDA::cudart m)
