cmake_minimum_required(VERSION 3.10)
project(roomsim_benchmark CUDA CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -std=c++20 --expt-relaxed-constexpr")

add_executable(roomsim roomsim.cpp)

# Treat .cpp file as CUDA source
set_source_files_properties(roomsim.cpp PROPERTIES LANGUAGE CUDA)

# Set CUDA architectures
set_target_properties(roomsim PROPERTIES CUDA_ARCHITECTURES "52;60;70;75;80;86")

# Enable optimizations and warnings
target_compile_options(roomsim PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3>)

# Link CUDA runtime and math library
target_link_libraries(roomsim PRIVATE ${CUDA_LIBRARIES} m)

# Include directories
target_include_directories(roomsim PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/..)



