cmake_minimum_required(VERSION 3.10)
if(POLICY CMP0104)
  cmake_policy(SET CMP0104 NEW)
endif()
project(floydwarshall_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

add_executable(floydwarshall floydwarshall.cpp)

# Compile the existing .cpp with NVCC (no new files)
set_source_files_properties(floydwarshall.cpp PROPERTIES LANGUAGE CUDA)

# Enable optimizations (per-language)
target_compile_options(floydwarshall PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:-O3 -march=native -Wall -Wextra>
  $<$<COMPILE_LANGUAGE:CUDA>:-O3>
)

# Prefer building device code for common modern architectures when supported
if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.18)
  set_property(TARGET floydwarshall PROPERTY CUDA_ARCHITECTURES 70 75 80 86)
endif()

# Link math library (CUDA runtime is linked automatically for CUDA targets)
target_link_libraries(floydwarshall m)
