cmake_minimum_required(VERSION 3.18)
project(cahn_hilliard_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES native)

set_source_files_properties(cahn_hilliard.cpp PROPERTIES LANGUAGE CUDA)
add_executable(cahn_hilliard cahn_hilliard.cpp)

# Enable optimizations
target_compile_options(cahn_hilliard PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:-O3 -Xcompiler=-march=native,-Wall,-Wextra>)

# Link math library
target_link_libraries(cahn_hilliard m)
