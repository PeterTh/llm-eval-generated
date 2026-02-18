cmake_minimum_required(VERSION 3.18)
project(roomsim_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find dependencies
find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)

add_executable(roomsim roomsim.cpp)

# Enable optimizations and warnings
target_compile_options(roomsim PRIVATE 
    $<$<COMPILE_LANGUAGE:CXX>:-O3 -march=native -Wall -Wextra -Wpedantic>
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 -arch=native>
)

# Use CUDA language for the source file to enable CUDA compilation
set_source_files_properties(roomsim.cpp PROPERTIES LANGUAGE CUDA)

# Link libraries
target_link_libraries(roomsim PRIVATE 
    MPI::MPI_CXX 
    OpenMP::OpenMP_CXX 
    m
)

# Allow CUDA to use C++20 features if supported
set_target_properties(roomsim PROPERTIES CUDA_STANDARD 20)
set_target_properties(roomsim PROPERTIES CUDA_STANDARD_REQUIRED ON)
