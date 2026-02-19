cmake_minimum_required(VERSION 3.18)

# Ensure CMake can find NVCC in environments where CUDA isn't on PATH.
set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc CACHE FILEPATH "" FORCE)

# Cluster-safe default; avoids "Failed to detect a default CUDA architecture" on headless nodes.
set(CMAKE_CUDA_ARCHITECTURES 70)

project(roomsim_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)
find_package(CUDAToolkit REQUIRED)

add_executable(roomsim roomsim.cpp)

# Compile roomsim.cpp with NVCC so we can embed CUDA kernels without adding new files.
set_source_files_properties(roomsim.cpp PROPERTIES LANGUAGE CUDA)
set_target_properties(roomsim PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

target_compile_options(roomsim PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:-O3;-Xcompiler=-O3;-Xcompiler=-march=native>
  $<$<COMPILE_LANGUAGE:CXX>:-O3;-march=native>
)

target_link_libraries(roomsim PRIVATE MPI::MPI_CXX OpenMP::OpenMP_CXX CUDA::cudart m)

# Ensure OpenMP is enabled for host compilation under NVCC.
target_compile_options(roomsim PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-fopenmp>)
