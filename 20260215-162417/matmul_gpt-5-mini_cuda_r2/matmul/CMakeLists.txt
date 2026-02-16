cmake_minimum_required(VERSION 3.10)
project(matmul_benchmark)

# Provide a reasonable default CUDA architecture and enable CUDA language after project
# This helps configure on systems where automatic detection may fail; change if needed.
set(CMAKE_CUDA_ARCHITECTURES 70)
enable_language(CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 14)

add_executable(matmul matmul.cpp)
# Compile matmul.cpp with NVCC (treat as CUDA source)
set_source_files_properties(matmul.cpp PROPERTIES LANGUAGE CUDA)
set_target_properties(matmul PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# Enable optimizations
# Use generator expressions to set flags per language
target_compile_options(matmul PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-O3 -march=native -Wall -Wextra>
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 -lineinfo -Xcompiler=-fPIC -arch=sm_70>
)

# Link math and CUDA runtime
find_package(CUDAToolkit QUIET)
if (CUDAToolkit_FOUND)
    target_link_libraries(matmul PRIVATE m CUDA::cudart)
else()
    target_link_libraries(matmul PRIVATE m cudart)
endif()
