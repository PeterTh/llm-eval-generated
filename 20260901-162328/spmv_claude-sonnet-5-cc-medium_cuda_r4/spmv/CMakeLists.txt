cmake_minimum_required(VERSION 3.18)
project(spmv_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES native)
endif()

add_executable(spmv spmv.cpp)

# Compile the single source file as CUDA so it can contain both host and
# device (__global__) code without needing a separate .cu file.
set_source_files_properties(spmv.cpp PROPERTIES LANGUAGE CUDA)

set_target_properties(spmv PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# Enable optimizations
target_compile_options(spmv PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:-O3>
)

# Link math library
target_link_libraries(spmv m)
