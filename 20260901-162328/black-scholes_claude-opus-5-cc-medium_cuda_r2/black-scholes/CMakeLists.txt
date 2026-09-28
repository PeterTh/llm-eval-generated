cmake_minimum_required(VERSION 3.18)
project(black_scholes_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES native)
endif()

# The benchmark source is CUDA; keep the original file name.
set_source_files_properties(black_scholes.cpp PROPERTIES LANGUAGE CUDA)

add_executable(black_scholes black_scholes.cpp)

# Enable optimizations. IEEE-compliant division/sqrt and no FMA contraction keep the
# device results as close as possible to the original scalar computation.
target_compile_options(black_scholes PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3;--prec-div=true;--prec-sqrt=true;--fmad=false;-Xcompiler=-O3,-Wall,-Wextra>
)

# Link math library
target_link_libraries(black_scholes m)
