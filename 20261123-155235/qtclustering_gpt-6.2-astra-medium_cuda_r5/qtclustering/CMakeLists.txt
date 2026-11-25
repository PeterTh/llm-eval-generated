cmake_minimum_required(VERSION 3.20)

# Prefer the active toolkit when a system nvcc is also on PATH.
if(NOT DEFINED CMAKE_CUDA_COMPILER AND NOT DEFINED ENV{CUDACXX}
   AND EXISTS "/usr/local/cuda/bin/nvcc")
    set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")
endif()
project(qtclustering LANGUAGES CUDA)

set_source_files_properties(qtclustering.cpp PROPERTIES LANGUAGE CUDA)
add_executable(qtclustering qtclustering.cpp)
set_target_properties(qtclustering PROPERTIES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON)
target_compile_options(qtclustering PRIVATE -O3 -Xcompiler=-march=native,-Wall,-Wextra)
target_link_libraries(qtclustering PRIVATE m)
