cmake_minimum_required(VERSION 3.24)
project(roomsim_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Target the GPU(s) present at build time unless overridden
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES native)
endif()

add_executable(roomsim roomsim.cpp)

# roomsim.cpp contains CUDA kernels; compile it with the CUDA compiler
set_source_files_properties(roomsim.cpp PROPERTIES LANGUAGE CUDA)

# Enable optimizations and warnings.
# --fmad=false and -ffp-contract=off keep floating-point results independent of
# fused multiply-add contraction, so host and device arithmetic round identically.
target_compile_options(roomsim PRIVATE
    -O3
    --fmad=false
    -Xcompiler -march=native
    -Xcompiler -Wall
    -Xcompiler -Wextra
    -Xcompiler -ffp-contract=off
)

# Link math and CUDA runtime libraries
target_link_libraries(roomsim m)
