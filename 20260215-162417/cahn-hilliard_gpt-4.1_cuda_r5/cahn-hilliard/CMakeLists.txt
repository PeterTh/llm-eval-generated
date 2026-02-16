cmake_minimum_required(VERSION 3.10)
project(cahn_hilliard_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(CUDA REQUIRED)

cuda_add_executable(cahn_hilliard cahn_hilliard.cpp)
set_source_files_properties(cahn_hilliard.cpp PROPERTIES LANGUAGE CUDA)

# Enable optimizations
set_target_properties(cahn_hilliard PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(cahn_hilliard PROPERTIES CUDA_ARCHITECTURES 60)
target_compile_options(cahn_hilliard PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3 -Xcompiler=-Wall,-Wextra>)

# Link math library
target_link_libraries(cahn_hilliard m)
