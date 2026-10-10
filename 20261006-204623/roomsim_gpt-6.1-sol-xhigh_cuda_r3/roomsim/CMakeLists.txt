cmake_minimum_required(VERSION 3.24)

# Compile the existing source as CUDA without changing its name. Respect an
# explicitly selected architecture; otherwise target the installed GPU.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES AND NOT DEFINED ENV{CUDAARCHS})
    set(CMAKE_CUDA_ARCHITECTURES native)
endif()
project(roomsim_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set_source_files_properties(roomsim.cpp PROPERTIES LANGUAGE CUDA)
add_executable(roomsim roomsim.cpp)

# Optimize both device and host code, retaining precise division/square roots
# and the original accumulation order (no fast-math reassociation).
target_compile_options(roomsim PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3;--fmad=true;-Xcompiler=-O3,-march=native,-Wall,-Wextra>)

# Link math library
target_link_libraries(roomsim m)
