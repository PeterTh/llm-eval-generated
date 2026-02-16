cmake_minimum_required(VERSION 3.10)
project(cahn_hilliard_benchmark)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

enable_language(CUDA)
add_executable(cahn_hilliard cahn_hilliard.cpp)
target_sources(cahn_hilliard PRIVATE cahn_hilliard.cpp)
set_target_properties(cahn_hilliard PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# Enable optimizations
target_compile_options(cahn_hilliard PRIVATE -O3 -arch=sm_60 -Xcompiler "-Wall -Wextra")

# Link math library
target_link_libraries(cahn_hilliard m)
