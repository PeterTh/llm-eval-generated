cmake_minimum_required(VERSION 3.10)

project(spmv_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)

add_executable(spmv spmv.cpp)

# Compile spmv.cpp with NVCC so we can embed CUDA kernels without adding new files.
set_source_files_properties(spmv.cpp PROPERTIES LANGUAGE CUDA)

if(CMAKE_VERSION VERSION_GREATER_EQUAL "3.18")
  set_property(TARGET spmv PROPERTY CUDA_ARCHITECTURES native)
endif()

target_compile_options(spmv PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:-O3 -march=native -Wall -Wextra>
  $<$<COMPILE_LANGUAGE:CUDA>:-O3 --use_fast_math -Xcompiler=-fopenmp -Xcompiler=-march=native -Xcompiler=-Wall -Xcompiler=-Wextra>
)

target_link_libraries(spmv PRIVATE MPI::MPI_CXX OpenMP::OpenMP_CXX m)
