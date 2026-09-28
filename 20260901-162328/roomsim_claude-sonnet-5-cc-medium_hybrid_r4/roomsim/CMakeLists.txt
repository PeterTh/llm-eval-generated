cmake_minimum_required(VERSION 3.18)
project(roomsim_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 86)
endif()

find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)
find_package(CUDAToolkit REQUIRED)

add_executable(roomsim roomsim.cpp)

# Compile the single translation unit as CUDA so host code, __global__/__device__
# kernels, MPI, and OpenMP pragmas can all coexist in one file.
set_source_files_properties(roomsim.cpp PROPERTIES LANGUAGE CUDA)

target_compile_options(roomsim PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3;--expt-relaxed-constexpr;-Xcompiler=-fopenmp,-O3,-Wall,-Wextra>
)

target_include_directories(roomsim PRIVATE ${MPI_CXX_INCLUDE_DIRS})

target_link_libraries(roomsim PRIVATE
    MPI::MPI_CXX
    OpenMP::OpenMP_CXX
    CUDA::curand
    CUDA::cudart
    m
)
