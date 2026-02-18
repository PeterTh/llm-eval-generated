cmake_minimum_required(VERSION 3.17)
project(roomsim_benchmark CUDA CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 -Xptxas -O3")

# Enable CUDA language
enable_language(CUDA)

add_executable(roomsim roomsim/roomsim.cpp)

# Set file as CUDA source
set_source_files_properties(roomsim/roomsim.cpp PROPERTIES LANGUAGE CUDA)

# Enable optimizations
target_compile_options(roomsim PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-O3 -march=native>)

# Link math library
target_link_libraries(roomsim m)
