cmake_minimum_required(VERSION 3.10)
project(cholesky_benchmark)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Enable CUDA support
enable_language(CUDA)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
# Target the available GPU architecture(s); fall back to a broad set
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 70 75 80 86 89 90)
endif()

# Mark the .cpp source to be compiled with nvcc (it contains CUDA kernels)
set_source_files_properties(cholesky.cpp PROPERTIES LANGUAGE CUDA)

add_executable(cholesky cholesky.cpp)

# Host compiler flags
set_target_properties(cholesky PROPERTIES
    CUDA_HOST_COMPILER ${CMAKE_CXX_COMPILER}
)

# Enable optimizations
target_compile_options(cholesky PRIVATE -O3 -march=native -Wall -Wextra)

# Link cuBLAS, cuSOLVER and math library
find_package(CUDAToolkit REQUIRED)
target_link_libraries(cholesky CUDA::cublas CUDA::cusolver m)
