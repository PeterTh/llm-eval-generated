cmake_minimum_required(VERSION 3.18)

project(qtclustering LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.24)
    set(CMAKE_CUDA_ARCHITECTURES native)
  else()
    set(CMAKE_CUDA_ARCHITECTURES 70 75 80 86 90)
  endif()
endif()

find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)

# The single translation unit contains CUDA kernels; compile it with nvcc.
set_source_files_properties(qtclustering.cpp PROPERTIES LANGUAGE CUDA)

add_executable(qtclustering qtclustering.cpp)

target_include_directories(qtclustering PRIVATE ${MPI_CXX_INCLUDE_DIRS})

target_compile_options(qtclustering PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:-O3>
  $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-O3,-march=native,-fopenmp,-Wall,-Wextra>
)

set_target_properties(qtclustering PROPERTIES CUDA_SEPARABLE_COMPILATION OFF)

target_link_libraries(qtclustering PRIVATE
  ${MPI_CXX_LIBRARIES}
  OpenMP::OpenMP_CXX
  m
)

target_link_options(qtclustering PRIVATE -fopenmp)
