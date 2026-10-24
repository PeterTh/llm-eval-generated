cmake_minimum_required(VERSION 3.24)
project(roomsim_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Build for the GPU(s) present on the build machine instead of the very old
# default architecture (which forces expensive PTX JIT at every startup).
set(CMAKE_CUDA_ARCHITECTURES native)

add_executable(roomsim roomsim.cpp)

# Compile the existing .cpp as CUDA so the GPU kernels live in the same file.
set_source_files_properties(roomsim.cpp PROPERTIES LANGUAGE CUDA)

# Enable optimizations and warnings. Disable FMA contraction in device code so
# that the arithmetic matches the sequential implementation.
target_compile_options(roomsim PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3;--fmad=false;-Xcompiler=-O3,-march=native,-Wall,-Wextra>
)

# Link math library and the CUDA runtime
target_link_libraries(roomsim m)
