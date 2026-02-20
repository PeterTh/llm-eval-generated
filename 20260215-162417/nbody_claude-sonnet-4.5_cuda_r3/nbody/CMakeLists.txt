cmake_minimum_required(VERSION 3.10)
project(nbody_benchmark CUDA CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Enable CUDA
enable_language(CUDA)

add_executable(nbody nbody.cu)

# Set CUDA architectures (targeting common GPUs)
set_target_properties(nbody PROPERTIES CUDA_ARCHITECTURES "60;70;75;80")

# Enable optimizations
target_compile_options(nbody PRIVATE 
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 -use_fast_math>
    $<$<COMPILE_LANGUAGE:CXX>:-O3 -march=native -Wall -Wextra>
)

# Link math library
target_link_libraries(nbody m)
