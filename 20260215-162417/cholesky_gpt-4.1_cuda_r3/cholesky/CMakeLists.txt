cmake_minimum_required(VERSION 3.10)
project(cholesky_benchmark)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

cmake_minimum_required(VERSION 3.10)
project(cholesky_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(cholesky cholesky.cpp)
set_target_properties(cholesky PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# Enable optimizations
target_compile_options(cholesky PRIVATE -O3 -Wall -Wextra)

# Link math library and CUDA
find_package(CUDAToolkit REQUIRED)
target_include_directories(cholesky PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
target_link_libraries(cholesky m CUDA::cudart)
