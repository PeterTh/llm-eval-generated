cmake_minimum_required(VERSION 3.10)
project(cholesky_benchmark)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Enable CUDA language
enable_language(CUDA)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Target GPU architecture (RTX 3090 = sm_86)
set(CMAKE_CUDA_ARCHITECTURES 86)

add_executable(cholesky cholesky.cpp)

# Compile .cpp with nvcc to support CUDA runtime API
set_source_files_properties(cholesky.cpp PROPERTIES LANGUAGE CUDA)

# Enable optimizations
set_target_properties(cholesky PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)
target_compile_options(cholesky PRIVATE -O3 -Xcompiler=-march=native -Wall -Wextra)

# Link CUDA runtime and cuSOLVER
find_package(CUDAToolkit REQUIRED)
target_link_libraries(cholesky PRIVATE CUDA::cudart CUDA::cusolver m)
