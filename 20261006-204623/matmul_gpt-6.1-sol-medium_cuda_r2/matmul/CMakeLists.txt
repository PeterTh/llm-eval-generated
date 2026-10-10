cmake_minimum_required(VERSION 3.18)
# Generate native GPU code when supported, while allowing explicit cross builds
# through -DCMAKE_CUDA_ARCHITECTURES or the CUDAARCHS environment variable.
if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.24 AND
   NOT DEFINED CMAKE_CUDA_ARCHITECTURES AND "$ENV{CUDAARCHS}" STREQUAL "")
    set(CMAKE_CUDA_ARCHITECTURES native)
endif()
project(matmul_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Compile the existing source with nvcc without renaming the executable or file.
set_source_files_properties(matmul.cpp PROPERTIES LANGUAGE CUDA)
add_executable(matmul matmul.cpp)

# CUDA runtime linkage is supplied by CMake's CUDA language support.
target_compile_options(matmul PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3;-Xcompiler=-O3,-Wall,-Wextra>)
target_link_libraries(matmul PRIVATE m)
