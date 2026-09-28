cmake_minimum_required(VERSION 3.18)
project(spmv_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.24)
    set(CMAKE_CUDA_ARCHITECTURES native)
  else()
    set(CMAKE_CUDA_ARCHITECTURES 70 80 86 90)
  endif()
endif()

find_package(MPI REQUIRED COMPONENTS CXX)
find_package(OpenMP REQUIRED COMPONENTS CXX)

add_executable(spmv spmv.cpp)

# The single translation unit contains CUDA kernels
set_source_files_properties(spmv.cpp PROPERTIES LANGUAGE CUDA)

set_target_properties(spmv PROPERTIES CUDA_SEPARABLE_COMPILATION OFF)

# Enable optimizations (host compiler flags have to be forwarded through nvcc)
target_compile_options(spmv PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:-O3>
  $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-O3,-march=native,-Wall,-Wextra,${OpenMP_CXX_FLAGS}>
)

target_include_directories(spmv PRIVATE ${MPI_CXX_INCLUDE_DIRS})
target_link_libraries(spmv PRIVATE ${MPI_CXX_LIBRARIES} ${OpenMP_CXX_LIBRARIES} m)
