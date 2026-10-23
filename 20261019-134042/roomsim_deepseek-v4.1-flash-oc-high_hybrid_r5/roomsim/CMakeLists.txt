cmake_minimum_required(VERSION 3.21)
project(roomsim_benchmark LANGUAGES CXX CUDA)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 86)
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)
find_package(CUDAToolkit REQUIRED)

add_executable(roomsim roomsim.cpp)
set_source_files_properties(roomsim.cpp PROPERTIES LANGUAGE CUDA)

target_compile_options(roomsim PRIVATE
    -O3
    $<$<COMPILE_LANGUAGE:CXX>:-march=native;-Wall;-Wextra;-Wpedantic>
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)

target_link_libraries(roomsim PRIVATE
    m
    MPI::MPI_CXX
    OpenMP::OpenMP_CXX
    CUDA::cudart)
