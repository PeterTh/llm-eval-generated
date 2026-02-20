cmake_minimum_required(VERSION 3.10)
project(roomsim_benchmark CUDA CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Find OpenMP
find_package(OpenMP REQUIRED)

# Treat .cpp as CUDA source
set_source_files_properties(roomsim.cpp PROPERTIES LANGUAGE CUDA)

add_executable(roomsim roomsim.cpp)

# Set CUDA architecture for maximum compatibility
set_target_properties(roomsim PROPERTIES
    CUDA_ARCHITECTURES "52;60;61;70;75;80;86"
    CUDA_SEPARABLE_COMPILATION ON
)

# Enable optimizations and warnings
target_compile_options(roomsim PRIVATE 
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 -Xcompiler -O3,-march=native,-Wall,-Wextra,-Wpedantic>
)

# Link libraries
target_link_libraries(roomsim PRIVATE m OpenMP::OpenMP_CXX cudart)
