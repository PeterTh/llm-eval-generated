cmake_minimum_required(VERSION 3.10)
project(roomsim_benchmark CUDA CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Enable CUDA
enable_language(CUDA)

# Set CUDA architectures for maximum compatibility and performance
set(CMAKE_CUDA_ARCHITECTURES 60-real 61-real 62-real 70-real 75-real 80-real 86-real 89-real 90-real)

add_executable(roomsim roomsim.cpp)

# Set CUDA properties
set_target_properties(roomsim PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_PROPAGATE_HOST_FLAGS ON
)

# Enable optimizations and warnings
target_compile_options(roomsim PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-O3 -march=native -Wall -Wextra -Wpedantic>)
target_compile_options(roomsim PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3 -Xptxas -O3>)

# Link math library and CUDA
target_link_libraries(roomsim m)

