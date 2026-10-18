cmake_minimum_required(VERSION 3.18)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 60 70 80 86)
endif()
project(roomsim_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set_source_files_properties(roomsim.cpp PROPERTIES LANGUAGE CUDA)
add_executable(roomsim roomsim.cpp)

# Enable optimizations and warnings
target_compile_options(roomsim PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3;--fmad=false;-Xcompiler=-O3,-march=native,-ffp-contract=off,-Wall,-Wextra>)

# Link math library
target_link_libraries(roomsim m)
