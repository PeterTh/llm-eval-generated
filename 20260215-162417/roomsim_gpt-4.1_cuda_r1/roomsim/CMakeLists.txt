cmake_minimum_required(VERSION 3.10)
project(roomsim_benchmark)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(roomsim roomsim.cpp)

# Enable optimizations and warnings
target_compile_options(roomsim PRIVATE -O3 -march=native -Wall -Wextra -Wpedantic)

# Enable CUDA
find_package(CUDA REQUIRED)
target_compile_options(roomsim PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>)
set_source_files_properties(roomsim.cpp PROPERTIES LANGUAGE CUDA)
target_link_libraries(roomsim m cuda)
