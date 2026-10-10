cmake_minimum_required(VERSION 3.24)

# Select the build machine's GPU unless the caller requests other architectures.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES native)
endif()
project(roomsim_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Keep the existing filename and executable name, but compile the source as CUDA.
set_source_files_properties(roomsim.cpp PROPERTIES LANGUAGE CUDA)
add_executable(roomsim roomsim.cpp)

# Ordered float accumulation and precise CUDA division/square root preserve the
# reference computation. Fast math would alter visibility and correlation ties.
target_compile_options(roomsim PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3;--fmad=false;-Xcompiler=-O3,-march=native,-Wall,-Wextra>
)
target_link_libraries(roomsim PRIVATE m)
