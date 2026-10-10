cmake_minimum_required(VERSION 3.18)
# Optimize for the installed GPU by default on CMake versions that support
# native detection; callers can still select deployment architectures explicitly.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES AND CMAKE_VERSION VERSION_GREATER_EQUAL 3.24)
    set(CMAKE_CUDA_ARCHITECTURES native)
endif()
project(cholesky_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Keep the existing source filename and executable name; compile it with nvcc.
set_source_files_properties(cholesky.cpp PROPERTIES LANGUAGE CUDA)
add_executable(cholesky cholesky.cpp)
target_compile_options(cholesky PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3;-Xcompiler=-Wall,-Wextra>)
# No vendor math libraries or other dependencies beyond the CUDA runtime.
target_link_libraries(cholesky PRIVATE m)
