cmake_minimum_required(VERSION 3.24)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES native)
endif()
project(roomsim_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(roomsim roomsim.cpp)
set_source_files_properties(roomsim.cpp PROPERTIES LANGUAGE CUDA)
set_target_properties(roomsim PROPERTIES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON)

# Preserve sequential floating-point accumulation (no fused multiply-add).
target_compile_options(roomsim PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3;--fmad=false;-Xcompiler=-Wall,-Wextra>)

# Link math library
target_link_libraries(roomsim m)
