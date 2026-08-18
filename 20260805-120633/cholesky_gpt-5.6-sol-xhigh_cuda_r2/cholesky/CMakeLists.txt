cmake_minimum_required(VERSION 3.24)

# The benchmark environment may pair a newer host compiler with the installed
# CUDA toolkit. This flag is needed while CMake identifies nvcc, before target
# options can be applied.
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --allow-unsupported-compiler")

project(cholesky_benchmark LANGUAGES CXX CUDA)

find_package(CUDAToolkit REQUIRED)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES native)

# Keep the existing source filename while compiling it as CUDA.
set_source_files_properties(cholesky.cpp PROPERTIES LANGUAGE CUDA)
add_executable(cholesky cholesky.cpp)

set_target_properties(cholesky PROPERTIES
    CUDA_SEPARABLE_COMPILATION OFF
    CUDA_RESOLVE_DEVICE_SYMBOLS OFF
)

target_compile_options(cholesky PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3;--extra-device-vectorization;-Xptxas=-O3;-Xcompiler=-Wall,-Wextra>
)

target_link_libraries(cholesky PRIVATE
    CUDA::cudart
    CUDA::cublas
    CUDA::cusolver
)
