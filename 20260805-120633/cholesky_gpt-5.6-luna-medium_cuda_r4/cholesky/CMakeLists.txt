cmake_minimum_required(VERSION 3.10)
if(POLICY CMP0104)
    cmake_policy(SET CMP0104 NEW)
endif()
# The benchmark environment uses an Ampere RTX 3090.  Generating native
# device code avoids carrying slower compatibility images in the executable.
set(CMAKE_CUDA_ARCHITECTURES 86 CACHE STRING "CUDA compute capability")

project(cholesky_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(CUDAToolkit REQUIRED)

add_executable(cholesky cholesky.cpp)
set_source_files_properties(cholesky.cpp PROPERTIES LANGUAGE CUDA)
set_target_properties(cholesky PROPERTIES CUDA_STANDARD 20
                      CUDA_STANDARD_REQUIRED ON)

# Enable optimizations
target_compile_options(cholesky PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-O3;-march=native;-Wall;-Wextra>
    $<$<COMPILE_LANGUAGE:CUDA>:-O3>)

# Link math library
target_link_libraries(cholesky PRIVATE CUDA::cudart m)
