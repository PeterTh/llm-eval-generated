cmake_minimum_required(VERSION 3.10)
project(nbody_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(CUDA REQUIRED)

add_executable(nbody nbody.cpp)
target_link_libraries(nbody PRIVATE cudart)
target_compile_options(nbody PRIVATE -O3 -Wall -Wextra)
set_target_properties(nbody PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_source_files_properties(nbody.cpp PROPERTIES LANGUAGE CUDA)
set(CMAKE_CUDA_ARCHITECTURES 60 CACHE STRING "CUDA architectures")

# Enable optimizations
target_compile_options(nbody PRIVATE -O3 -Wall -Wextra)

# Link math library
target_link_libraries(nbody PRIVATE m cudart)
