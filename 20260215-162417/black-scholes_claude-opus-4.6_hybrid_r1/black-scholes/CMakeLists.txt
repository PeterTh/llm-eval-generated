cmake_minimum_required(VERSION 3.18)
project(black_scholes_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 86)
endif()

find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)

set_source_files_properties(black_scholes.cpp PROPERTIES LANGUAGE CUDA)

add_executable(black_scholes black_scholes.cpp)

target_compile_options(black_scholes PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-fopenmp,-O3,-march=native>
)

target_include_directories(black_scholes PRIVATE ${MPI_CXX_INCLUDE_DIRS})
target_link_libraries(black_scholes PRIVATE ${MPI_CXX_LIBRARIES} m)
target_link_options(black_scholes PRIVATE -fopenmp)
