cmake_minimum_required(VERSION 3.24)

# CUDA 12.0 predates GCC 13.  This is harmless with supported host compilers
# and lets CMake's CUDA compiler-identification step work on current systems.
set(CMAKE_CUDA_FLAGS_INIT "-allow-unsupported-compiler")
project(unstructured_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES native)

find_package(CUDAToolkit REQUIRED)
find_package(MPI REQUIRED COMPONENTS CXX)
find_package(OpenMP REQUIRED COMPONENTS CXX)

add_executable(unstructured unstructured.cpp)

# Keep the requested executable name while compiling the existing source as
# CUDA: NVCC compiles its host portions with OpenMP and links MPI/CUDA runtime.
set_source_files_properties(unstructured.cpp PROPERTIES LANGUAGE CUDA)
set_target_properties(unstructured PROPERTIES
    CUDA_SEPARABLE_COMPILATION OFF
    CUDA_RESOLVE_DEVICE_SYMBOLS OFF)

target_compile_options(unstructured PRIVATE
    -O3
    -Wall
    -Wextra
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-fopenmp>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-march=native>)

target_link_libraries(unstructured PRIVATE
    CUDA::cudart
    MPI::MPI_CXX
    OpenMP::OpenMP_CXX
    m)
