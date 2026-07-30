cmake_minimum_required(VERSION 3.10)
project(matmul_benchmark)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Enable CUDA language
enable_language(CUDA)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Find CUDA toolkit for include dirs and libraries
find_package(CUDAToolkit REQUIRED)

add_executable(matmul matmul.cpp)

# Force compilation with nvcc (CUDA compiler)
set_source_files_properties(matmul.cpp PROPERTIES LANGUAGE CUDA)

# Set CUDA architecture for RTX 3090 (compute capability 8.6)
set_target_properties(matmul PROPERTIES CUDA_ARCHITECTURES "86")

# Enable optimizations
target_compile_options(matmul PRIVATE -O3 -Wall -Wextra)
target_compile_options(matmul PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:-O3>")

# Link CUDA runtime and math library
target_link_libraries(matmul ${CUDAToolkit_LIBRARIES} m)
target_include_directories(matmul PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
