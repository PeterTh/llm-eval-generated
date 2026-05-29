cmake_minimum_required(VERSION 3.10)
project(nbody_benchmark)

enable_language(CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Target RTX 3090 compute capability
set(CMAKE_CUDA_ARCHITECTURES 86)

add_executable(nbody nbody.cpp)
set_source_files_properties(nbody.cpp PROPERTIES LANGUAGE CUDA)

# Enable optimizations
target_compile_options(nbody PRIVATE -O3 -Wall -Wextra)

# Link math library
target_link_libraries(nbody m)

# Include common headers
target_include_directories(nbody PRIVATE ../common)
