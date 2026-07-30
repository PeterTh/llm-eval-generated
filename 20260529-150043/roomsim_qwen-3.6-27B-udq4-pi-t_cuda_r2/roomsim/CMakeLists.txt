cmake_minimum_required(VERSION 3.10)
project(roomsim_benchmark)

enable_language(CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Set CUDA architecture for maximum performance
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 80)
endif()

add_executable(roomsim roomsim.cpp)

# Force compilation as CUDA (file contains CUDA kernels)
set_source_files_properties(roomsim.cpp PROPERTIES LANGUAGE CUDA)

set_target_properties(roomsim PROPERTIES
    CUDA_SEPARABLE_COMPILATION OFF
)

# Enable optimizations and warnings
target_compile_options(roomsim PRIVATE
    -O3 -Xcompiler=-march=native -Xcompiler=-Wall -Xcompiler=-Wextra
)

# Link math library
target_link_libraries(roomsim m)
