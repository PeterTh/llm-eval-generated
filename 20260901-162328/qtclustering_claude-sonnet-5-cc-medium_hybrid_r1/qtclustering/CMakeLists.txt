cmake_minimum_required(VERSION 3.18)

project(qtclustering LANGUAGES CXX CUDA)

find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)
find_package(CUDAToolkit REQUIRED)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Compile the (single) source file as CUDA so it can hold both host
# (MPI/OpenMP) code and device (CUDA) kernels together.
set_source_files_properties(qtclustering.cpp PROPERTIES LANGUAGE CUDA)

add_executable(qtclustering qtclustering.cpp)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES "80;86")
endif()
set_target_properties(qtclustering PROPERTIES
    CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
    CUDA_SEPARABLE_COMPILATION ON
)

target_compile_options(qtclustering PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 --extended-lambda -Xcompiler=-fopenmp,-O3,-march=native,-Wall,-Wextra>
)

target_include_directories(qtclustering PRIVATE ${MPI_CXX_INCLUDE_DIRS})

target_link_libraries(qtclustering PRIVATE
    MPI::MPI_CXX
    OpenMP::OpenMP_CXX
    CUDA::cudart
    m
)
