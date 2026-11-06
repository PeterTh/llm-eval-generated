cmake_minimum_required(VERSION 3.18)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES AND NOT DEFINED ENV{CUDAARCHS})
    set(_matmul_default_cuda_architectures ON)
endif()
project(matmul_benchmark LANGUAGES CXX CUDA)

# Include native code for common GPUs and PTX for later architectures.  A
# caller can override this with CMAKE_CUDA_ARCHITECTURES or CUDAARCHS.
if(_matmul_default_cuda_architectures AND CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 13.0)
    set(CMAKE_CUDA_ARCHITECTURES "75-real;80-real;86" CACHE STRING "CUDA architectures" FORCE)
elseif(_matmul_default_cuda_architectures AND CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 11.1)
    set(CMAKE_CUDA_ARCHITECTURES "52-real;60-real;70-real;75-real;80-real;86" CACHE STRING "CUDA architectures" FORCE)
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(matmul matmul.cpp)
set_source_files_properties(matmul.cpp PROPERTIES LANGUAGE CUDA)
set_target_properties(matmul PROPERTIES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON)

# Enable optimizations
target_compile_options(matmul PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3>)

# Link math library
target_link_libraries(matmul m)
