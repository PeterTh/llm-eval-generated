cmake_minimum_required(VERSION 3.10)
project(roomsim_benchmark)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

enable_language(CUDA)

add_executable(roomsim roomsim.cpp)
set_target_properties(roomsim PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_source_files_properties(roomsim.cpp PROPERTIES LANGUAGE CUDA)

# Enable optimizations and warnings
target_compile_options(roomsim PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3 -arch=sm_60> -O3 -march=native -Wall -Wextra -Wpedantic)

# Link math library
target_link_libraries(roomsim m)
