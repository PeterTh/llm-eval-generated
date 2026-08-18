cmake_minimum_required(VERSION 3.18)
project(cahn_hilliard_benchmark LANGUAGES CXX CUDA)

find_package(MPI REQUIRED COMPONENTS CXX)
find_package(OpenMP REQUIRED COMPONENTS CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Keep the requested executable name. Compiling the existing source as CUDA lets
# MPI host orchestration and accelerator kernels remain in one translation unit.
set_source_files_properties(cahn_hilliard.cpp PROPERTIES LANGUAGE CUDA)
add_executable(cahn_hilliard cahn_hilliard.cpp)

target_link_libraries(cahn_hilliard PRIVATE MPI::MPI_CXX OpenMP::OpenMP_CXX)
target_compile_options(cahn_hilliard PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3;--expt-relaxed-constexpr;-Xcompiler=-Wall,-Wextra,-fopenmp>
    $<$<COMPILE_LANGUAGE:CXX>:-O3;-Wall;-Wextra>)

# Emit native SASS for the accelerator generations supported by the installed
# toolkit instead of relying on forward-compatible (and slower) cubins.
set(CH_CUDA_ARCHITECTURES 70 75 80)
if(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 11.1)
    list(APPEND CH_CUDA_ARCHITECTURES 86)
endif()
if(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 11.8)
    list(APPEND CH_CUDA_ARCHITECTURES 89)
endif()
if(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 12.0)
    list(APPEND CH_CUDA_ARCHITECTURES 90)
endif()
set_target_properties(cahn_hilliard PROPERTIES
    CUDA_SEPARABLE_COMPILATION OFF
    CUDA_ARCHITECTURES "${CH_CUDA_ARCHITECTURES}")
