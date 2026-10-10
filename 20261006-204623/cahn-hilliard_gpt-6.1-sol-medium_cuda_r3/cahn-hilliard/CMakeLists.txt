cmake_minimum_required(VERSION 3.24)
# Target installed GPUs by default; cross builds can specify architectures.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES native)
endif()
project(cahn_hilliard_benchmark LANGUAGES CXX CUDA)

# Compile the existing source as CUDA without changing the executable name.
set_source_files_properties(cahn_hilliard.cpp PROPERTIES LANGUAGE CUDA)
add_executable(cahn_hilliard cahn_hilliard.cpp)
set_target_properties(cahn_hilliard PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON)

# Preserve the original double-precision expression ordering. No fast math.
target_compile_options(cahn_hilliard PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3;--fmad=false;-Xcompiler=-Wall,-Wextra>)
