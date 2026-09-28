cmake_minimum_required(VERSION 3.18)
project(nbody_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.24)
    set(CMAKE_CUDA_ARCHITECTURES native)
  else()
    set(CMAKE_CUDA_ARCHITECTURES 70 80 86)
  endif()
endif()

find_package(MPI REQUIRED COMPONENTS CXX)
find_package(OpenMP REQUIRED COMPONENTS CXX)

# The single translation unit contains CUDA kernels and is compiled by nvcc.
set_source_files_properties(nbody.cpp PROPERTIES LANGUAGE CUDA)

add_executable(nbody nbody.cpp)

# Enable optimizations (host flags are forwarded to the host compiler by nvcc)
target_compile_options(nbody PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:-O3
    --prec-div=true --prec-sqrt=true
    -Xptxas=-O3
    -Xcompiler=-O3
    -Xcompiler=-march=native
    -Xcompiler=${OpenMP_CXX_FLAGS}
    -Xcompiler=-Wall
    -Xcompiler=-Wextra>)

target_include_directories(nbody PRIVATE ${MPI_CXX_INCLUDE_DIRS})
target_link_options(nbody PRIVATE ${OpenMP_CXX_FLAGS})
target_link_libraries(nbody PRIVATE ${MPI_CXX_LIBRARIES} ${OpenMP_CXX_LIBRARIES} m)
