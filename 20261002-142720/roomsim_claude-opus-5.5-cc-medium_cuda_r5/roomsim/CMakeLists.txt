cmake_minimum_required(VERSION 3.18)

# Build for the GPU(s) of the build machine unless specified otherwise
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.24)
        set(CMAKE_CUDA_ARCHITECTURES native)
    else()
        set(CMAKE_CUDA_ARCHITECTURES 86)
    endif()
endif()

project(roomsim_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(Threads REQUIRED)

# The single source file is compiled as CUDA
set_source_files_properties(roomsim.cpp PROPERTIES LANGUAGE CUDA)
add_executable(roomsim roomsim.cpp)

# Enable optimizations and warnings.
# Device code: no automatic FMA contraction; the kernels spell out the exact
# rounding (FMA vs. separate mul/add) of the host reference build.
# Host code: GNU-mode floating-point contraction like the original gnu++20 build.
target_compile_options(roomsim PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 -fmad=false -Xcompiler=-O3,-march=native,-ffp-contract=fast,-Wall,-Wextra>)

# Link math library and threads
target_link_libraries(roomsim Threads::Threads m)
