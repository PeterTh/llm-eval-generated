cmake_minimum_required(VERSION 3.18)
project(cahn_hilliard_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Target the GPU(s) present on the build machine for maximum performance.
# A specific architecture can still be supplied with -DCMAKE_CUDA_ARCHITECTURES.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES native)
endif()

add_executable(cahn_hilliard cahn_hilliard.cpp)

# The benchmark keeps its original .cpp file name; compile it with nvcc as CUDA.
set_source_files_properties(cahn_hilliard.cpp PROPERTIES LANGUAGE CUDA)

# Enable optimizations
target_compile_options(cahn_hilliard PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3;-Xcompiler=-O3;-Xcompiler=-Wall,-Wextra;-x;cu>
    $<$<COMPILE_LANGUAGE:CXX>:-O3;-march=native;-Wall;-Wextra>
)

# Link math library
target_link_libraries(cahn_hilliard m)
