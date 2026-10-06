cmake_minimum_required(VERSION 3.18)

# nvcc 12.0 does not accept GCC 13 as host compiler; prefer GCC 12 when available
if(NOT DEFINED CMAKE_CUDA_HOST_COMPILER AND NOT DEFINED ENV{CUDAHOSTCXX})
    find_program(GXX12 g++-12)
    if(GXX12)
        set(CMAKE_CUDA_HOST_COMPILER ${GXX12})
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

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# The single source file is compiled as CUDA
set_source_files_properties(roomsim.cpp PROPERTIES LANGUAGE CUDA)
add_executable(roomsim roomsim.cpp)

# Enable optimizations and warnings. FP contraction is disabled on host and device so
# that the GPU results are bit-reproducible IEEE single-precision arithmetic.
target_compile_options(roomsim PRIVATE
    -O3 --fmad=false
    -Xcompiler=-O3,-march=native,-ffp-contract=off,-Wall,-Wextra)

# Link math and thread libraries (host threads produce the RNG snapshots)
find_package(Threads REQUIRED)
target_link_libraries(roomsim m Threads::Threads)
