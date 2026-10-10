cmake_minimum_required(VERSION 3.24)

# Prefer the selected CUDA installation, while respecting compiler overrides.
if(NOT CMAKE_CUDA_COMPILER AND NOT DEFINED ENV{CUDACXX})
    find_program(ROOMSIM_NVCC nvcc HINTS /usr/local/cuda/bin)
    if(ROOMSIM_NVCC)
        set(CMAKE_CUDA_COMPILER "${ROOMSIM_NVCC}" CACHE FILEPATH "CUDA compiler")
    endif()
endif()
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES native)
endif()
project(roomsim_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Compile the existing source as CUDA without renaming it.
set_source_files_properties(roomsim.cpp PROPERTIES LANGUAGE CUDA)
add_executable(roomsim roomsim.cpp)

# Use fused arithmetic as in the optimized host build, without fast-math.
# The kernels preserve the original floating-point accumulation order.
target_compile_options(roomsim PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3;--fmad=true;-Xcompiler=-O3,-march=native,-Wall,-Wextra>)

# Link math library
target_link_libraries(roomsim m)
