cmake_minimum_required(VERSION 3.10)

project(nbody_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)

add_executable(nbody nbody.cpp)

# Build the existing .cpp as CUDA to enable kernels in the same translation unit.
set_source_files_properties(nbody.cpp PROPERTIES LANGUAGE CUDA)

# Enable optimizations
# Note: this target is compiled as CUDA (nvcc), so pass OpenMP flags through to the host compiler.
target_compile_options(nbody PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:-O3 -march=native -Wall -Wextra>
  $<$<COMPILE_LANGUAGE:CUDA>:-O3 -Xcompiler=-fopenmp -Xcompiler=-march=native -Xcompiler=-Wall -Xcompiler=-Wextra>
)

# Prefer compiling for the local accelerator for maximum performance.
set_property(TARGET nbody PROPERTY CUDA_ARCHITECTURES native)

target_link_libraries(nbody PRIVATE MPI::MPI_CXX OpenMP::OpenMP_CXX m)
