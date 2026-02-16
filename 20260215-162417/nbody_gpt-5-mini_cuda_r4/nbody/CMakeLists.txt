cmake_minimum_required(VERSION 3.10)
project(nbody_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Compile the single source file with NVCC by forcing CUDA language
set_source_files_properties(nbody.cpp PROPERTIES LANGUAGE CUDA)

add_executable(nbody nbody.cpp)

# Enable optimizations for C++ and CUDA
target_compile_options(nbody PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-O3 -march=native -Wall -Wextra>
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 -Xcompiler=-march=native -Xcompiler=-Wall -Xcompiler=-Wextra>
)

# Ensure CUDA uses C++20 when compiling device code
set_property(TARGET nbody PROPERTY CUDA_STANDARD 14)
set_property(TARGET nbody PROPERTY CUDA_STANDARD_REQUIRED ON)

# Link math library
target_link_libraries(nbody m)
