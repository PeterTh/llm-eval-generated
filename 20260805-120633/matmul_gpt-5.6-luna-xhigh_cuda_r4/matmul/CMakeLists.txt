cmake_minimum_required(VERSION 3.18)

# The benchmark environment uses the RTX 3090 (compute capability 8.6).
# Defining this before enabling CUDA prevents CMake's older default from
# selecting a lower-performance architecture; it remains overridable with
# -DCMAKE_CUDA_ARCHITECTURES=<arch>.
set(CMAKE_CUDA_ARCHITECTURES 86 CACHE STRING "CUDA architectures to compile for")

project(matmul_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Keep the existing source file and executable name while compiling the source
# as CUDA so the benchmark always executes the GPU implementation.
set_source_files_properties(matmul.cpp PROPERTIES LANGUAGE CUDA)
add_executable(matmul matmul.cpp)

# Enable optimizations
target_compile_options(matmul PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall,-Wextra>
)

# Link math library
target_link_libraries(matmul PRIVATE m)
