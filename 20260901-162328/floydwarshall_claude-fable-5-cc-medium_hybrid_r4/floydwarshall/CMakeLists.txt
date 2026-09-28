cmake_minimum_required(VERSION 3.18)
project(floydwarshall_benchmark LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# nvcc requires a supported GNU host compiler (<= gcc 12 for CUDA 12.0).
if(NOT DEFINED CMAKE_CUDA_HOST_COMPILER)
    foreach(_gxx g++-12 g++-11)
        find_program(_cuda_host_gxx ${_gxx})
        if(_cuda_host_gxx)
            set(CMAKE_CUDA_HOST_COMPILER ${_cuda_host_gxx})
            break()
        endif()
    endforeach()
endif()

enable_language(CUDA)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES OR CMAKE_CUDA_ARCHITECTURES STREQUAL "")
    set(CMAKE_CUDA_ARCHITECTURES 70 80 86)
endif()

find_package(MPI REQUIRED COMPONENTS CXX)
find_package(OpenMP REQUIRED COMPONENTS CXX)

add_executable(floydwarshall floydwarshall.cpp)

# The single source file contains CUDA kernels; compile it with nvcc.
set_source_files_properties(floydwarshall.cpp PROPERTIES LANGUAGE CUDA)

# Enable optimizations (host-side flags forwarded through nvcc)
target_compile_options(floydwarshall PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 -Xcompiler=-O3,-march=native,-Wall,-Wextra,-fopenmp>)

target_include_directories(floydwarshall PRIVATE ${MPI_CXX_INCLUDE_DIRS})
target_link_libraries(floydwarshall
    ${MPI_CXX_LIBRARIES}
    OpenMP::OpenMP_CXX
    m)
