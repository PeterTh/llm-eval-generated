cmake_minimum_required(VERSION 3.10)
project(cahn_hilliard_benchmark LANGUAGES CXX CUDA)

if(POLICY CMP0104)
  cmake_policy(SET CMP0104 NEW)
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

add_executable(cahn_hilliard cahn_hilliard.cpp)

# Compile the existing .cpp with NVCC (no new files).
set_source_files_properties(cahn_hilliard.cpp PROPERTIES LANGUAGE CUDA)

# Build for the local GPU when supported.
if(CMAKE_VERSION VERSION_GREATER_EQUAL "3.24")
  set_property(TARGET cahn_hilliard PROPERTY CUDA_ARCHITECTURES native)
elseif(CMAKE_VERSION VERSION_GREATER_EQUAL "3.18")
  set_property(TARGET cahn_hilliard PROPERTY CUDA_ARCHITECTURES 75)
endif()

# Enable optimizations
target_compile_options(cahn_hilliard PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:-O3;--fmad=false;-Xcompiler=-O3,-march=native,-Wall,-Wextra>
)

# Link math library
target_link_libraries(cahn_hilliard m)
