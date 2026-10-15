cmake_minimum_required(VERSION 3.18)
project(cholesky_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 80 86)
endif()

add_executable(cholesky cholesky.cpp)
set_source_files_properties(cholesky.cpp PROPERTIES LANGUAGE CUDA)
set_target_properties(cholesky PROPERTIES CUDA_SEPARABLE_COMPILATION OFF)

# Enable optimizations
target_compile_options(cholesky PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3 -Xcompiler=-Wall,-Wextra>)
set_property(TARGET cholesky PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)

target_link_libraries(cholesky m)
