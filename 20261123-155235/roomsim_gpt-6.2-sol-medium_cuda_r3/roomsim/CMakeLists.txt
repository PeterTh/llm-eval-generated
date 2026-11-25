cmake_minimum_required(VERSION 3.20)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES AND "$ENV{CUDAARCHS}" STREQUAL "")
    set(roomsim_auto_cuda_arch ON)
endif()

project(roomsim_benchmark LANGUAGES CXX CUDA)
if(roomsim_auto_cuda_arch AND CMAKE_CUDA_ARCHITECTURES_NATIVE)
    set(CMAKE_CUDA_ARCHITECTURES native CACHE STRING "CUDA architectures for roomsim" FORCE)
endif()

# Keep the benchmark in its existing source file.
set_source_files_properties(roomsim.cpp PROPERTIES LANGUAGE CUDA)
add_executable(roomsim roomsim.cpp)
set_target_properties(roomsim PROPERTIES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON)

target_compile_options(roomsim PRIVATE -O3 --fmad=false
    -Xcompiler=-march=native,-Wall,-Wextra)
