cmake_minimum_required(VERSION 3.10)
project(nbody_benchmark LANGUAGES CXX C CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)

add_executable(nbody nbody.cpp)

set_source_files_properties(nbody.cpp PROPERTIES LANGUAGE CUDA)

# Enable optimizations and CUDA support
if(CMAKE_CUDA_COMPILER_ID STREQUAL "NVIDIA")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode arch=compute_70,code=sm_70 -gencode arch=compute_80,code=sm_80")
endif()

target_compile_options(nbody PRIVATE 
    $<$<COMPILE_LANGUAGE:CXX>:-O3 -march=native -Wall -Wextra>
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 -Xcompiler=-O3,-Wall,-Wextra,-fopenmp>
)

target_link_libraries(nbody PRIVATE MPI::MPI_CXX OpenMP::OpenMP_CXX m)

set_target_properties(nbody PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
