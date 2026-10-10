cmake_minimum_required(VERSION 3.24)

# Compile for the installed GPU unless an architecture was explicitly selected.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES native)
endif()
project(cholesky_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Keep the existing source file and executable names.
set_source_files_properties(cholesky.cpp PROPERTIES LANGUAGE CUDA)
add_executable(cholesky cholesky.cpp)

# Enable optimizations
target_compile_options(cholesky PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-O3,-march=native,-Wall,-Wextra>)

# Link math library
target_link_libraries(cholesky m)
