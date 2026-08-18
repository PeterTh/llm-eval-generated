cmake_minimum_required(VERSION 3.18)
project(roomsim_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Compile the existing translation unit as CUDA.  "native" keeps the generated
# device code tuned to the GPU used for the benchmark while retaining the
# required single executable name.
set(CMAKE_CUDA_ARCHITECTURES native)

add_executable(roomsim roomsim.cpp)
set_source_files_properties(roomsim.cpp PROPERTIES LANGUAGE CUDA)

# Enable optimizations and host warnings without fast-math, preserving the
# original floating-point semantics as closely as CUDA permits.
target_compile_options(roomsim PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-O3;-march=native;-Wall;-Wextra;-Wpedantic>
    $<$<COMPILE_LANGUAGE:CUDA>:-O3;--expt-relaxed-constexpr>)

# Link math library
target_link_libraries(roomsim m)
