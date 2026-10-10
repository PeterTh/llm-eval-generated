cmake_minimum_required(VERSION 3.24)
project(black_scholes_benchmark LANGUAGES CXX)

# CUDA 12.0 does not support GCC 13, which may be the system default.
# Respect explicit host compiler selections and prefer a supported GCC if needed.
if(NOT DEFINED CMAKE_CUDA_HOST_COMPILER AND NOT DEFINED ENV{CUDAHOSTCXX}
   AND CMAKE_CXX_COMPILER_ID STREQUAL "GNU"
   AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 13)
    find_program(BLACK_SCHOLES_CUDA_HOST_COMPILER NAMES g++-12)
    if(BLACK_SCHOLES_CUDA_HOST_COMPILER)
        set(CMAKE_CUDA_HOST_COMPILER "${BLACK_SCHOLES_CUDA_HOST_COMPILER}")
    endif()
endif()

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES native)
endif()
enable_language(CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Keep the existing source filename while compiling its kernels with nvcc.
set_source_files_properties(black_scholes.cpp PROPERTIES LANGUAGE CUDA)
add_executable(black_scholes black_scholes.cpp)

# Preserve double precision and expression rounding; do not enable fast math.
target_compile_options(black_scholes PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3;--fmad=false;-Xcompiler=-O3,-march=native,-Wall,-Wextra>
)

# Link math library
target_link_libraries(black_scholes m)
