cmake_minimum_required(VERSION 3.10)
project(cahn_hilliard_benchmark CUDA CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Prefer the fastest architecture detected; fall back to compute_50
if(NOT CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES OFF)
endif()

add_executable(cahn_hilliard cahn_hilliard.cpp)

# Force the .cpp source through nvcc so CUDA runtime APIs are available
set_source_files_properties(cahn_hilliard.cpp PROPERTIES LANGUAGE CUDA)

# CUDA compile options: max performance
set_target_properties(cahn_hilliard PROPERTIES
    CUDA_SEPARABLE_COMPILATION OFF
)
target_compile_options(cahn_hilliard PRIVATE
    -O3
    -use_fast_math
    -Xcompiler=-Wall
    -Xcompiler=-Wextra
)

# Link math library
target_link_libraries(cahn_hilliard m)
