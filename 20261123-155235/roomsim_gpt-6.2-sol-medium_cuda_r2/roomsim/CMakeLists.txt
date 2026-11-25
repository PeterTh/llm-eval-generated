cmake_minimum_required(VERSION 3.24)

# Use the build machine's GPU when CMake can detect it. Allow explicit
# architecture selection for cross-compilation and GPU-free build machines.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES AND "$ENV{CUDAARCHS}" STREQUAL "")
    set(roomsim_detect_architecture ON)
endif()

project(roomsim_benchmark LANGUAGES CUDA)

if(roomsim_detect_architecture AND CMAKE_CUDA_ARCHITECTURES_NATIVE)
    set(CMAKE_CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES_NATIVE}"
        CACHE STRING "CUDA architectures" FORCE)
endif()

set_source_files_properties(roomsim.cpp PROPERTIES LANGUAGE CUDA)
add_executable(roomsim roomsim.cpp)
set_target_properties(roomsim PROPERTIES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON)
target_compile_options(roomsim PRIVATE -O3 -Xcompiler=-O3,-march=native,-Wall,-Wextra)
