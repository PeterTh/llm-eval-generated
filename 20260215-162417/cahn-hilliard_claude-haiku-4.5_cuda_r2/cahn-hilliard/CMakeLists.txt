cmake_minimum_required(VERSION 3.10)
project(cahn_hilliard_benchmark CUDA CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Set CUDA architectures
set(CMAKE_CUDA_ARCHITECTURES 70 75 80)

# Enable CUDA code compilation for .cpp files
set_source_files_properties(cahn_hilliard.cpp PROPERTIES LANGUAGE CUDA)

# Set CUDA compilation flags with optimizations
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 -Xcompiler -O3 -Xcompiler -march=native")

add_executable(cahn_hilliard cahn_hilliard.cpp)

# Set output directory to parent build directory
set_target_properties(cahn_hilliard PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")

# Enable optimizations for CXX
target_compile_options(cahn_hilliard PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-O3 -march=native -Wall -Wextra>)

# Link math library and CUDA runtime libraries
target_link_libraries(cahn_hilliard m cudart)
