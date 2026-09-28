cmake_minimum_required(VERSION 3.20)
project(roomsim_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Build for the GPU(s) present in this machine (fall back to a wide set).
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES native)
endif()

# The benchmark is a single translation unit that is compiled as CUDA.
set_source_files_properties(roomsim.cpp PROPERTIES LANGUAGE CUDA)

add_executable(roomsim roomsim.cpp)

# Enable optimizations and warnings.  Fast-math is deliberately NOT enabled so
# that the GPU results stay faithful to the reference implementation.
target_compile_options(roomsim PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:-O3
    --extended-lambda
    --expt-relaxed-constexpr
    -Xcompiler=-O3,-march=native,-Wall,-Wextra>)

set_target_properties(roomsim PROPERTIES CUDA_SEPARABLE_COMPILATION OFF)

# Link math library
target_link_libraries(roomsim m)
