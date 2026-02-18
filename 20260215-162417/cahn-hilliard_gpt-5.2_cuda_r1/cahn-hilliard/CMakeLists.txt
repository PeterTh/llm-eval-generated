cmake_minimum_required(VERSION 3.10)

# Prefer a known NVCC location when CUDA isn't on PATH.
if(NOT DEFINED CMAKE_CUDA_COMPILER AND EXISTS "/usr/local/cuda/bin/nvcc")
  set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc" CACHE FILEPATH "CUDA compiler" FORCE)
endif()

project(cahn_hilliard_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

add_executable(cahn_hilliard cahn_hilliard.cpp)

# Compile the existing .cpp file as CUDA (no new source files).
set_source_files_properties(cahn_hilliard.cpp PROPERTIES LANGUAGE CUDA)

# Enable optimizations
target_compile_options(cahn_hilliard PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:-O3 -march=native -Wall -Wextra>
  $<$<COMPILE_LANGUAGE:CUDA>:-O3 --fmad=false --prec-div=true --prec-sqrt=true --ftz=false>
)

set_target_properties(cahn_hilliard PROPERTIES CUDA_SEPARABLE_COMPILATION OFF)

# Link math library
target_link_libraries(cahn_hilliard PRIVATE m)
