cmake_minimum_required(VERSION 3.18)

# CUDA 12 requires a supported host compiler. Respect explicit toolchain choices.
if(NOT DEFINED CMAKE_CUDA_HOST_COMPILER AND NOT DEFINED ENV{CUDAHOSTCXX})
    find_program(ROOMSIM_CUDA_HOST_COMPILER NAMES g++-12 g++-11)
    if(ROOMSIM_CUDA_HOST_COMPILER)
        set(CMAKE_CUDA_HOST_COMPILER "${ROOMSIM_CUDA_HOST_COMPILER}" CACHE FILEPATH "CUDA host compiler")
    endif()
endif()
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES AND NOT DEFINED ENV{CUDAARCHS})
    set(CMAKE_CUDA_ARCHITECTURES 60 70 75 80 86)
endif()
project(roomsim_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Keep the existing source filename and executable name.
set_source_files_properties(roomsim.cpp PROPERTIES LANGUAGE CUDA)
add_executable(roomsim roomsim.cpp)
# Retain precise division/square roots and ordered sums. Host geometry uses
# the same native optimizations as the original benchmark.
target_compile_options(roomsim PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3;--fmad=true;-Xcompiler=-O3,-march=native,-Wall,-Wextra>)
target_link_libraries(roomsim PRIVATE m)
