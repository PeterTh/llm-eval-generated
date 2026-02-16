cmake_minimum_required(VERSION 3.10)
project(nbody_benchmark)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(CUDA REQUIRED)

cuda_add_executable(nbody nbody.cpp)

# Enable optimizations
target_compile_options(nbody PRIVATE -O3 -Wall -Wextra)
target_compile_options(nbody PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-arch=sm_60>)

# Link math library
target_link_libraries(nbody m)
