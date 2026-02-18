cmake_minimum_required(VERSION 3.18)
project(nbody_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES 86)

find_package(MPI REQUIRED)

set_source_files_properties(nbody.cpp PROPERTIES LANGUAGE CUDA)

add_executable(nbody nbody.cpp)

target_include_directories(nbody PRIVATE ${MPI_CXX_INCLUDE_DIRS})

target_compile_options(nbody PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 -Xcompiler=-O3,-march=native,-fopenmp,-Wall,-Wextra>
)

target_link_libraries(nbody ${MPI_CXX_LIBRARIES} gomp m)
