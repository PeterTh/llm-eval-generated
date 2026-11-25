cmake_minimum_required(VERSION 3.18)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES AND "$ENV{CUDAARCHS}" STREQUAL "")
    set(roomsim_auto_architecture ON)
endif()
project(roomsim_benchmark LANGUAGES CUDA)
if(roomsim_auto_architecture AND CMAKE_CUDA_ARCHITECTURES_NATIVE)
    set(CMAKE_CUDA_ARCHITECTURES native CACHE STRING "CUDA architectures for roomsim" FORCE)
endif()

# Compile the existing source file as CUDA without changing its name.
set_source_files_properties(roomsim.cpp PROPERTIES LANGUAGE CUDA)
add_executable(roomsim roomsim.cpp)
set_target_properties(roomsim PROPERTIES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON)
target_compile_options(roomsim PRIVATE -O3 -Xcompiler=-O3,-march=native,-Wall,-Wextra)
