cmake_minimum_required(VERSION 3.10)
project(matmul_benchmark)

enable_language(CUDA)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Use compute capability 8.6 (RTX 30-series)
set(CMAKE_CUDA_ARCHITECTURES 86)

add_executable(matmul matmul.cpp)
set_source_files_properties(matmul.cpp PROPERTIES LANGUAGE CUDA)

# Enable optimizations
target_compile_options(matmul PRIVATE -O3 -Wall -Wextra)

# Link CUDA runtime and math library
find_library(CUDA_CUDART_LIBRARY cudart HINTS ${CUDA_TOOLKIT_ROOT_DIR}/lib64 ${CUDA_TOOLKIT_ROOT_DIR}/lib)
target_link_libraries(matmul ${CUDA_CUDART_LIBRARY} m)
