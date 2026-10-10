cmake_minimum_required(VERSION 3.24)
# Tune for the available GPU unless the caller requests other architectures.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES AND NOT DEFINED ENV{CUDAARCHS})
    set(CMAKE_CUDA_ARCHITECTURES native)
endif()
project(cholesky_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set_source_files_properties(cholesky.cpp PROPERTIES LANGUAGE CUDA)
add_executable(cholesky cholesky.cpp)

# Keep double-precision arithmetic; CUDA's default fused operations are safe
# here, but fast-math approximations would change the numerical guarantees.
target_compile_options(cholesky PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3;-Xcompiler=-O3,-Wall,-Wextra>
)

# CUDA runtime linkage is supplied by the CUDA language; no BLAS dependency.
target_link_libraries(cholesky m)
