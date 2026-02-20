cmake_minimum_required(VERSION 3.10)
project(roomsim_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

add_executable(roomsim roomsim.cpp)

# Compile the existing .cpp as CUDA to enable GPU kernels without renaming files.
set_source_files_properties(roomsim.cpp PROPERTIES LANGUAGE CUDA)

# Enable optimizations and warnings
target_compile_options(roomsim PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:-O3 -march=native -Wall -Wextra -Wpedantic>
  $<$<COMPILE_LANGUAGE:CUDA>:-O3 --fmad=false>
)

# Reasonable default set for forward/backward compatibility across common GPUs.
if (CMAKE_VERSION VERSION_GREATER_EQUAL 3.18)
  set_property(TARGET roomsim PROPERTY CUDA_ARCHITECTURES 75 80 86 90)
endif()

# Link math library
target_link_libraries(roomsim m)
