cmake_minimum_required(VERSION 3.17)

project(nbody_benchmark)

# Enable CUDA and CXX languages
enable_language(CUDA)
enable_language(CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Set reasonable CUDA architectures for compatibility
if(NOT CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 52 60 70 75 80)
endif()

add_executable(nbody nbody.cpp)

# Set CUDA as C++ compiler wrapper to handle CUDA files
set_source_files_properties(nbody.cpp PROPERTIES LANGUAGE CUDA)

# Enable optimizations
target_compile_options(nbody PRIVATE -O3 -march=native -Wall -Wextra)
target_compile_options(nbody PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3 -use_fast_math>)

# Link math library and CUDA runtime
target_link_libraries(nbody m)
