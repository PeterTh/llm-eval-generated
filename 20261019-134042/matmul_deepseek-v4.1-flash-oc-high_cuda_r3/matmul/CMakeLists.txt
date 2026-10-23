cmake_minimum_required(VERSION 3.18)

# Build for the architecture of the GPU present on the build machine.  This
# must be set before project() enables CUDA, otherwise CMake keeps the CUDA
# compiler's default (sm_52), which is much slower on newer GPUs.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES native CACHE STRING "CUDA architectures" FORCE)
endif()

project(matmul_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

add_executable(matmul matmul.cpp)

# The benchmark implementation is CUDA, so compile the existing source as CUDA.
set_source_files_properties(matmul.cpp PROPERTIES LANGUAGE CUDA)

# Enable optimizations for both the device code and the host portion that nvcc
# compiles from the same translation unit.
target_compile_options(matmul PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3>
    $<$<COMPILE_LANGUAGE:CUDA>:--fmad=false>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-O3>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-march=native>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wextra>
)

# Link math library
target_link_libraries(matmul PRIVATE m)
