cmake_minimum_required(VERSION 3.18)
project(cahn_hilliard_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

add_executable(cahn_hilliard cahn_hilliard.cpp)

# The single translation unit contains CUDA device code and therefore must be
# compiled by nvcc even though it uses the .cpp extension.
set_source_files_properties(cahn_hilliard.cpp PROPERTIES LANGUAGE CUDA)

# Compile for the GPUs present on the build host so the benchmark runs natively
# on the local accelerator instead of relying on PTX JIT compilation.
if(CMAKE_VERSION VERSION_GREATER_EQUAL "3.24")
    set_target_properties(cahn_hilliard PROPERTIES CUDA_ARCHITECTURES native)
else()
    set_target_properties(cahn_hilliard PROPERTIES CUDA_ARCHITECTURES "61;70;75;80;86")
endif()

# Enable optimizations
target_compile_options(cahn_hilliard PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-O3 -march=native -Wall -Wextra>
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 -Xcompiler=-Wall,-Wextra>)

# Link math library
target_link_libraries(cahn_hilliard m)
