cmake_minimum_required(VERSION 3.10)
project(nbody_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

add_executable(nbody nbody.cpp)

# Enable optimizations
if(CMAKE_CUDA_COMPILER_ID STREQUAL "NVIDIA")
    set_source_files_properties(nbody.cpp PROPERTIES LANGUAGE CUDA)
    target_compile_options(nbody PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3 -use_fast_math -arch=native>)
else()
    target_compile_options(nbody PRIVATE -O3 -march=native -Wall -Wextra)
endif()

# Link math library
target_link_libraries(nbody m)
