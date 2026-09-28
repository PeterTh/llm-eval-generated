cmake_minimum_required(VERSION 3.18)
project(roomsim_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES native)
endif()

# roomsim.cpp contains CUDA kernels and is compiled by nvcc
set_source_files_properties(roomsim.cpp PROPERTIES LANGUAGE CUDA)

add_executable(roomsim roomsim.cpp)

set_target_properties(roomsim PROPERTIES CUDA_SEPARABLE_COMPILATION OFF)

# Enable optimizations and warnings (host code is compiled through nvcc)
target_compile_options(roomsim PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:-O3>
  $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>
  $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-O3,-march=native,-fopenmp,-Wall,-Wextra>
)

target_include_directories(roomsim PRIVATE ${MPI_CXX_INCLUDE_DIRS})
target_link_libraries(roomsim ${MPI_CXX_LIBRARIES} OpenMP::OpenMP_CXX m)
