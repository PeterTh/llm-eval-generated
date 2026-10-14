cmake_minimum_required(VERSION 3.24)
project(nbody_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Build for the GPU architecture(s) present on the machine
set(CMAKE_CUDA_ARCHITECTURES native)

add_executable(nbody nbody.cpp)

# nbody.cpp contains CUDA device code, so compile it with nvcc
set_source_files_properties(nbody.cpp PROPERTIES LANGUAGE CUDA)

# Enable optimizations
target_compile_options(nbody PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-O3 -march=native -Wall -Wextra>
    # --fmad=false keeps each multiply and add rounded separately, matching the source semantics
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 --fmad=false -Xcompiler=-Wall,-Wextra,-march=native>)

# Link math library
target_link_libraries(nbody m)
