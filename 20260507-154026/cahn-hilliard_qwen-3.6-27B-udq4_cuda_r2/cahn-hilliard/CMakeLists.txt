cmake_minimum_required(VERSION 3.10)
project(cahn_hilliard_benchmark CUDA CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

enable_language(CUDA)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES 86)

add_executable(cahn_hilliard cahn_hilliard.cu)

# Enable optimizations
set_target_properties(cahn_hilliard PROPERTIES
    CUDA_ARCHITECTURES "86"
)
target_compile_options(cahn_hilliard PRIVATE -O3 -Xcompiler=-march=native,-Wall,-Wextra)

# Link math library
target_link_libraries(cahn_hilliard m)
