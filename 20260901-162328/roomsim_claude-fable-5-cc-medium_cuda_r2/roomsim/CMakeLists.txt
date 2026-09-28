cmake_minimum_required(VERSION 3.18)
project(roomsim_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 86)
endif()

add_executable(roomsim roomsim.cpp)

# The whole translation unit is CUDA (host + device code)
set_source_files_properties(roomsim.cpp PROPERTIES LANGUAGE CUDA)

# Enable optimizations and warnings
target_compile_options(roomsim PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:-O3 --expt-relaxed-constexpr -Xcompiler=-O3,-march=native,-Wall,-Wextra>)

# Link math library
target_link_libraries(roomsim m)
