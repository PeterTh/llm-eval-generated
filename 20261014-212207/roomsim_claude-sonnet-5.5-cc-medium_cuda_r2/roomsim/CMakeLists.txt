cmake_minimum_required(VERSION 3.18)
project(roomsim_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 75 80 86 89)
endif()

add_executable(roomsim roomsim.cpp)

# The source is CUDA (host + device code in one translation unit)
set_source_files_properties(roomsim.cpp PROPERTIES LANGUAGE CUDA)

# -fmad=false keeps device arithmetic bit-compatible with the sequential reference
target_compile_options(roomsim PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 -fmad=false -lineinfo
        -Xcompiler=-O3,-march=native,-Wall,-Wextra>)

set_target_properties(roomsim PROPERTIES CUDA_SEPARABLE_COMPILATION OFF)

# Link math library
target_link_libraries(roomsim m)
