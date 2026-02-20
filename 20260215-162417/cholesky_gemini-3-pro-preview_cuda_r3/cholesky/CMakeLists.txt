cmake_minimum_required(VERSION 3.10)
project(cholesky_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(cholesky cholesky.cpp)

# Enable optimizations
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(cholesky PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-O3 -march=native -Wall -Wextra>)
endif()

# Link math library
target_link_libraries(cholesky PRIVATE m)

# Link CUDA libraries
find_package(CUDAToolkit REQUIRED)
target_link_libraries(cholesky PRIVATE CUDA::cudart CUDA::cusolver CUDA::cublas)

set_source_files_properties(cholesky.cpp PROPERTIES LANGUAGE CUDA)
