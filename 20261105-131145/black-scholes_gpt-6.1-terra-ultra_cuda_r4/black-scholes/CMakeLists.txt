cmake_minimum_required(VERSION 3.10)
if(POLICY CMP0104)
    cmake_policy(SET CMP0104 NEW)
endif()
project(black_scholes_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set_source_files_properties(black_scholes.cpp PROPERTIES LANGUAGE CUDA)
add_executable(black_scholes black_scholes.cpp)
if(POLICY CMP0104 AND NOT CMAKE_CUDA_ARCHITECTURES)
    # An existing build directory may predate CMake's architecture detection.
    set_property(TARGET black_scholes PROPERTY CUDA_ARCHITECTURES OFF)
endif()

# Enable optimizations
target_compile_options(black_scholes PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3;--fmad=false;-Xcompiler=-Wall,-Wextra>)

# Link math library
target_link_libraries(black_scholes m)
