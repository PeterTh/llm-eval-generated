cmake_minimum_required(VERSION 3.24)
project(black_scholes_benchmark LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(MPI REQUIRED COMPONENTS CXX)
find_package(OpenMP REQUIRED COMPONENTS CXX)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES native)
endif()

enable_language(CUDA)

find_package(CUDAToolkit REQUIRED)

set_source_files_properties(black_scholes.cpp PROPERTIES LANGUAGE CUDA)

add_executable(black_scholes black_scholes.cpp)

target_compile_options(black_scholes PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-O3>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-march=native>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-fopenmp>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall,-Wextra>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xptxas=-O3>
)

target_link_libraries(black_scholes PRIVATE
    MPI::MPI_CXX
    CUDA::cudart
    ${OpenMP_CXX_LIBRARIES}
    m
)
