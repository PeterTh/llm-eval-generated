cmake_minimum_required(VERSION 3.18)

# Target the available NVIDIA architecture (set before project() so that the
# CUDA toolchain picks it up instead of the generic default).
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 86)
endif()

project(roomsim_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Compile the existing source as CUDA so kernels can live in the same file.
set_source_files_properties(roomsim.cpp PROPERTIES LANGUAGE CUDA)

add_executable(roomsim roomsim.cpp)

# Enable optimizations for device code (host optimization comes from the
# build type).  Do not enable fast math: the reference uses IEEE semantics.
target_compile_options(roomsim PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 --extra-device-vectorization>)

# Link math and CUDA runtime libraries
target_link_libraries(roomsim m)
