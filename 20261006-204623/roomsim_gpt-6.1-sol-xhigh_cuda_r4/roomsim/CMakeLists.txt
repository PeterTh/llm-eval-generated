cmake_minimum_required(VERSION 3.24)

# Prefer a host compiler supported by CUDA 12, while honoring toolchain choices.
if(NOT DEFINED CMAKE_CUDA_HOST_COMPILER AND NOT DEFINED ENV{CUDAHOSTCXX})
    find_program(ROOMSIM_CUDA_HOST_COMPILER NAMES g++-12 g++-11)
    if(ROOMSIM_CUDA_HOST_COMPILER)
        set(CMAKE_CUDA_HOST_COMPILER "${ROOMSIM_CUDA_HOST_COMPILER}")
    endif()
endif()

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES native)
endif()
project(roomsim_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set_source_files_properties(roomsim.cpp PROPERTIES LANGUAGE CUDA)
add_executable(roomsim roomsim.cpp)

# Retain the original accumulation order and allow IEEE fused multiply-add.
target_compile_options(roomsim PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3;--fmad=true;-Xcompiler=-O3,-march=native,-Wall,-Wextra>)

# Link math library
target_link_libraries(roomsim m)
