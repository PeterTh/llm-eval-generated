cmake_minimum_required(VERSION 3.10)
project(roomsim_benchmark CUDA CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Enable CUDA
enable_language(CUDA)

add_executable(roomsim roomsim.cu)

# Set CUDA architectures for compatibility
set_target_properties(roomsim PROPERTIES CUDA_ARCHITECTURES "50;60;70;80;90")

# Enable optimizations and warnings
target_compile_options(roomsim PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-O3 -march=native -Wall -Wextra -Wpedantic>)
target_compile_options(roomsim PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3>)

# Link math library and CUDA runtime
target_link_libraries(roomsim m)
