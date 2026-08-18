cmake_minimum_required(VERSION 3.24)

# CUDA 12.0 predates GCC 13, which is the system compiler on some benchmark
# hosts.  The generated host code only uses the ordinary, supported C++ ABI.
set(CMAKE_CUDA_FLAGS_INIT "--allow-unsupported-compiler")
project(unstructured_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Keep the original filename and executable name while compiling the translation
# unit with nvcc so that host and device code can remain together.
set_source_files_properties(unstructured.cpp PROPERTIES LANGUAGE CUDA)
add_executable(unstructured unstructured.cpp)

# Native device code avoids PTX JIT overhead and lets ptxas tune scheduling and
# occupancy for the GPU on which this benchmark is built.
set_target_properties(unstructured PROPERTIES
    CUDA_ARCHITECTURES native
    CUDA_SEPARABLE_COMPILATION OFF
)

target_compile_options(unstructured PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3;--extra-device-vectorization;-Xptxas=-O3;-Xcompiler=-O3,-march=native,-Wall,-Wextra>
)
