cmake_minimum_required(VERSION 3.24)
# Build for the installed GPU unless the caller supplies target architectures.
# Select this before enabling CUDA, which otherwise initializes a generic default.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES AND "$ENV{CUDAARCHS}" STREQUAL "")
    set(CMAKE_CUDA_ARCHITECTURES native)
endif()
project(cahn_hilliard_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Keep the existing source filename and executable name.
set_source_files_properties(cahn_hilliard.cpp PROPERTIES LANGUAGE CUDA)
add_executable(cahn_hilliard cahn_hilliard.cpp)

# Preserve double-precision expression order (no fast math or FMA contraction).
target_compile_options(cahn_hilliard PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3;--fmad=false;-Xcompiler=-O3,-Wall,-Wextra>)
target_link_libraries(cahn_hilliard PRIVATE m)
