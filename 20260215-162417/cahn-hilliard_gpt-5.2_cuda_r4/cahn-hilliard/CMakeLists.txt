cmake_minimum_required(VERSION 3.10)
project(cahn_hilliard_benchmark LANGUAGES CXX CUDA)

# Prefer native GPU codegen when supported by the active CMake.
if(POLICY CMP0104)
    cmake_policy(SET CMP0104 NEW)
endif()
if(NOT CMAKE_CUDA_ARCHITECTURES)
    # CMake >= 3.18 honors this; with newer CMake it avoids JIT overhead.
    set(CMAKE_CUDA_ARCHITECTURES native)
endif()

# Keep the code portable across NVCC versions.
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

add_executable(cahn_hilliard cahn_hilliard.cpp)

# Compile the existing .cpp as CUDA (no new files / same executable name).
set_source_files_properties(cahn_hilliard.cpp PROPERTIES LANGUAGE CUDA)
set_property(TARGET cahn_hilliard PROPERTY CUDA_SEPARABLE_COMPILATION ON)

# Enable optimizations
target_compile_options(cahn_hilliard PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-O3 -march=native -Wall -Wextra>
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 -lineinfo>
)

# Link math library
target_link_libraries(cahn_hilliard m)
