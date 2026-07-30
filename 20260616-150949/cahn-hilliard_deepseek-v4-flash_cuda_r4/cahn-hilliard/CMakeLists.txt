cmake_minimum_required(VERSION 3.18)
project(cahn_hilliard_benchmark LANGUAGES CUDA CXX)

set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(cahn_hilliard cahn_hilliard.cpp)

# Compile the .cpp file as CUDA
set_source_files_properties(cahn_hilliard.cpp PROPERTIES LANGUAGE CUDA)

# Target CUDA architectures (Volta through Hopper)
set_target_properties(cahn_hilliard PROPERTIES
    CUDA_ARCHITECTURES "70;75;80;86;89"
)

# Enable optimizations
target_compile_options(cahn_hilliard PRIVATE -O3 -Xcompiler=-march=native -Xcompiler=-Wall -Xcompiler=-Wextra)

# Link math library
target_link_libraries(cahn_hilliard PRIVATE m)
