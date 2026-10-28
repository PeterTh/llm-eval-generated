cmake_minimum_required(VERSION 3.18)

# Target GPU architectures (override with -DSPMV_CUDA_ARCHITECTURES=...).
# Must be set before project() so CMake does not replace it with the compiler default.
set(SPMV_CUDA_ARCHITECTURES "70;75;80;86;89;90" CACHE STRING "CUDA architectures to build for")
set(CMAKE_CUDA_ARCHITECTURES ${SPMV_CUDA_ARCHITECTURES})

project(spmv_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(CUDAToolkit REQUIRED)

# The benchmark source is CUDA-enabled; compile it with nvcc
set_source_files_properties(spmv.cpp PROPERTIES LANGUAGE CUDA)

add_executable(spmv spmv.cpp)

# Enable optimizations (host flags only for the C++ compiler, device flags for nvcc)
target_compile_options(spmv PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:-O3 -march=native -Wall -Wextra>
  $<$<COMPILE_LANGUAGE:CUDA>:-O3 -Xcompiler=-Wall,-Wextra>
)

# Link math library and CUDA runtime
target_link_libraries(spmv m CUDA::cudart)
