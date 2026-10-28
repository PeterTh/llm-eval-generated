cmake_minimum_required(VERSION 3.18)
project(roomsim_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Target the architectures of the installed GPU(s) when supported by CMake
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.24)
    set(CMAKE_CUDA_ARCHITECTURES native)
  else()
    set(CMAKE_CUDA_ARCHITECTURES 75 80 86 89)
  endif()
endif()

add_executable(roomsim roomsim.cpp)

# The benchmark source contains CUDA kernels; compile it with nvcc.
set_source_files_properties(roomsim.cpp PROPERTIES LANGUAGE CUDA)

# Enable optimizations and warnings
target_compile_options(roomsim PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:-O3>
  $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-march=native>
  $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall>
  $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wextra>
)

# Link math library
target_link_libraries(roomsim m)
