cmake_minimum_required(VERSION 3.10)
project(cahn_hilliard_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Rename .cpp to .cu for CUDA compilation
set_source_files_properties(cahn_hilliard.cpp PROPERTIES LANGUAGE CUDA)

add_executable(cahn_hilliard cahn_hilliard.cpp)

# Enable optimizations for both host and device code
target_compile_options(cahn_hilliard PRIVATE 
    $<$<COMPILE_LANGUAGE:CXX>:-O3 -march=native -Wall -Wextra>
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 --use_fast_math -Xcompiler=-Wall,-Wextra>
)

# Set CUDA architectures for better performance
set_target_properties(cahn_hilliard PROPERTIES 
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "60;70;75;80;86;89;90"
)

# Link math library
target_link_libraries(cahn_hilliard m)
