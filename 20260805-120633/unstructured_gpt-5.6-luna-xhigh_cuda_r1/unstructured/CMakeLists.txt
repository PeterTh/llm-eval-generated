cmake_minimum_required(VERSION 3.10)
project(unstructured_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# The benchmark runs on the Ampere GPUs used by this benchmark environment.
# Keeping the architecture explicit avoids compiling unused device code and
# gives nvcc the opportunity to optimize for the actual execution hardware.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 86)
endif()

# Keep the original .cpp filename while compiling it as CUDA.  This preserves
# the executable name and the existing source layout.
set_source_files_properties(unstructured.cpp PROPERTIES LANGUAGE CUDA)

add_executable(unstructured unstructured.cpp)

# Enable optimizations.  FMA contraction is disabled for the device code so
# the ordered floating-point operations retain the reference semantics.
target_compile_options(unstructured PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-O3;-march=native;-Wall;-Wextra>
    $<$<COMPILE_LANGUAGE:CUDA>:-O3;--fmad=false;-Xcompiler=-Wall,-Wextra>
)

# Link math library
target_link_libraries(unstructured m)
