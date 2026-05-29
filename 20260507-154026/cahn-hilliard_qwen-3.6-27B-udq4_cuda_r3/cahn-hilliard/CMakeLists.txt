cmake_minimum_required(VERSION 3.17)
project(cahn_hilliard_benchmark CUDA CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

enable_language(CUDA)
set(CMAKE_CUDA_ARCHITECTURES OFF)

add_executable(cahn_hilliard cahn_hilliard.cpp)
set_source_files_properties(cahn_hilliard.cpp PROPERTIES LANGUAGE CUDA)

# Enable optimizations
target_compile_options(cahn_hilliard PRIVATE -O3 -march=native -Wall -Wextra)
set_target_properties(cahn_hilliard PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# Link CUDA runtime and math library
find_library(CUDART_LIBRARY cudart PATHS /usr/lib/x86_64-linux-gnu /usr/local/cuda/lib64)
target_link_libraries(cahn_hilliard ${CUDART_LIBRARY} m)
