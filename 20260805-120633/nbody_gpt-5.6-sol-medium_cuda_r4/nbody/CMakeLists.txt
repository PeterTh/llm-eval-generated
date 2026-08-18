cmake_minimum_required(VERSION 3.24)
project(nbody_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(nbody nbody.cpp)
set_source_files_properties(nbody.cpp PROPERTIES LANGUAGE CUDA)
set_target_properties(nbody PROPERTIES
    CUDA_ARCHITECTURES native
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON
)

# Build host support code and GPU kernels with their respective optimizing
# compilers.  Device linking is unnecessary because all CUDA code is in one TU.
target_compile_options(nbody PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-O3;-march=native;-Wall;-Wextra>
    $<$<COMPILE_LANGUAGE:CUDA>:-O3;--extra-device-vectorization;-Xcompiler=-Wall,-Wextra>
)

# Link math library
target_link_libraries(nbody m)
