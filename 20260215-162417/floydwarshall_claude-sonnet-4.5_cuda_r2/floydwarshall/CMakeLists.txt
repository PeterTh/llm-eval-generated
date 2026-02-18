cmake_minimum_required(VERSION 3.18)
project(floydwarshall_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Rename the .cpp to .cu for CUDA compilation
set_source_files_properties(floydwarshall.cpp PROPERTIES LANGUAGE CUDA)

add_executable(floydwarshall floydwarshall.cpp)

# Enable optimizations
target_compile_options(floydwarshall PRIVATE 
    $<$<COMPILE_LANGUAGE:CXX>:-O3 -march=native -Wall -Wextra>
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 -lineinfo>
)

# Set CUDA architectures for better performance
set_target_properties(floydwarshall PROPERTIES 
    CUDA_ARCHITECTURES "70;75;80;86;89;90"
    CUDA_SEPARABLE_COMPILATION ON
)

# Link math library
target_link_libraries(floydwarshall m)

