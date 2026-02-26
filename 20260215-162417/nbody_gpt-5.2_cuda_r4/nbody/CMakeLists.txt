cmake_minimum_required(VERSION 3.18)
project(nbody_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Build for the current GPU by default (max performance on the target machine)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES native)
endif()

add_executable(nbody nbody.cpp)

# Compile the existing .cpp as CUDA (no new files / same executable name)
set_source_files_properties(nbody.cpp PROPERTIES LANGUAGE CUDA)
set_target_properties(nbody PROPERTIES CUDA_SEPARABLE_COMPILATION OFF)

target_compile_options(nbody PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:-O3;-Xptxas=-O3;-Xcompiler=-O3,-march=native,-Wall,-Wextra>
)

# Link math library (kept for compatibility)
target_link_libraries(nbody m)
