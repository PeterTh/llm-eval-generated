cmake_minimum_required(VERSION 3.18)

# Prefer nvcc if present; fall back to clang++ CUDA if nvcc isn't on PATH.
if(NOT DEFINED CMAKE_CUDA_COMPILER)
  find_program(_NVCC nvcc HINTS /usr/local/cuda/bin /usr/local/cuda-*/bin)
  if(_NVCC)
    set(CMAKE_CUDA_COMPILER "${_NVCC}" CACHE FILEPATH "" FORCE)
  else()
    find_program(_CLANG_CUDA clang++ HINTS /usr/bin /usr/local/bin)
    if(_CLANG_CUDA)
      set(CMAKE_CUDA_COMPILER "${_CLANG_CUDA}" CACHE FILEPATH "" FORCE)
    endif()
  endif()
endif()

# CMake cannot always auto-detect a default GPU arch in containerized/CI environments.
# Build fatbins for common datacenter GPUs.
set(CMAKE_CUDA_ARCHITECTURES 60 70 80 CACHE STRING "" FORCE)

project(cholesky_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)
find_package(CUDAToolkit REQUIRED)

add_executable(cholesky cholesky.cpp)

# Compile the single translation unit with NVCC so CUDA kernels can live in cholesky.cpp.
set_source_files_properties(cholesky.cpp PROPERTIES LANGUAGE CUDA)

# Optimizations
target_compile_options(cholesky PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:-O3 -march=native -Wall -Wextra>
  $<$<COMPILE_LANGUAGE:CUDA>:-O3 -lineinfo>
)

target_link_libraries(cholesky PRIVATE MPI::MPI_CXX OpenMP::OpenMP_CXX CUDA::cudart m)
