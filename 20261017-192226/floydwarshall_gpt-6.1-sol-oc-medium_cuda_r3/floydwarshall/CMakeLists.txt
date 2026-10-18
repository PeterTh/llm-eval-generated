cmake_minimum_required(VERSION 3.18)
# Target the installed GPU by default, avoiding generic PTX JIT compilation.
# Explicit architecture selections remain available for cross-compilation.
if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.24 AND
   NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES native CACHE STRING "CUDA target architectures")
endif()
project(floydwarshall_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set_source_files_properties(floydwarshall.cpp PROPERTIES LANGUAGE CUDA)
add_executable(floydwarshall floydwarshall.cpp)

# Enable optimizations
target_compile_options(floydwarshall PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3;-Xcompiler=-Wall,-Wextra>)

# Link math library
target_link_libraries(floydwarshall m)
