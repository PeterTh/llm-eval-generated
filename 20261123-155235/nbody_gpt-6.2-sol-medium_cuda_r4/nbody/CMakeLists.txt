cmake_minimum_required(VERSION 3.20)

# Include device binaries for each supported GPU generation by default.
if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.23 AND
   NOT DEFINED CMAKE_CUDA_ARCHITECTURES AND NOT DEFINED ENV{CUDAARCHS})
    set(CMAKE_CUDA_ARCHITECTURES all-major)
endif()

project(nbody_benchmark LANGUAGES CXX CUDA)

set_source_files_properties(nbody.cpp PROPERTIES LANGUAGE CUDA)
add_executable(nbody nbody.cpp)
set_target_properties(nbody PROPERTIES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON)

target_compile_options(nbody PRIVATE -O3 -Xcompiler=-O3,-march=native,-Wall,-Wextra)
