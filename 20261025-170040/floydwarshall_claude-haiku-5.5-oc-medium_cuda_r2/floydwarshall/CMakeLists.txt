cmake_minimum_required(VERSION 3.24)

# Generate device code for common modern GPU architectures unless the user overrides it
# with -DCMAKE_CUDA_ARCHITECTURES=... (must be set before project(): CMake otherwise
# pre-populates the variable with the nvcc default, which is sm_52 on some toolkits).
if(NOT CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 75 80 86 89)
endif()

project(floydwarshall_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(CUDAToolkit REQUIRED)

add_executable(floydwarshall floydwarshall.cpp)

# The implementation is CUDA; compile the .cpp source with nvcc
set_source_files_properties(floydwarshall.cpp PROPERTIES LANGUAGE CUDA)

# Enable optimizations (host flags forwarded through nvcc)
target_compile_options(floydwarshall PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 -Xcompiler=-march=native -Xcompiler=-Wall -Xcompiler=-Wextra>)

# Link CUDA runtime and math library
target_link_libraries(floydwarshall PRIVATE CUDA::cudart m)
