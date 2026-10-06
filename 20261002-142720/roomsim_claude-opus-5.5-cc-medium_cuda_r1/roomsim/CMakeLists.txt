cmake_minimum_required(VERSION 3.18)

# Build for the GPUs of this machine unless architectures are given explicitly
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.24)
        set(CMAKE_CUDA_ARCHITECTURES native)
    else()
        set(CMAKE_CUDA_ARCHITECTURES 70 75 80 86)
    endif()
endif()

project(roomsim_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# The benchmark source contains the CUDA kernels
set_source_files_properties(roomsim.cpp PROPERTIES LANGUAGE CUDA)

add_executable(roomsim roomsim.cpp)

# Device code: no implicit FMA contraction (fused operations are explicit so
# that results match the host reference bit-for-bit).
# Host code: same optimization and FP-contraction behaviour as the reference build.
target_compile_options(roomsim PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 -fmad=false
        -Xcompiler=-O3,-march=native,-ffp-contract=fast,-Wall,-Wextra>)

# Link math and thread libraries (one host thread drives each GPU)
find_package(Threads REQUIRED)
target_link_libraries(roomsim m Threads::Threads)
