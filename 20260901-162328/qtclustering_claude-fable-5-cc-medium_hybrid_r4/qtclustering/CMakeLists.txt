cmake_minimum_required(VERSION 3.18)

project(qtclustering LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(NOT CMAKE_CUDA_ARCHITECTURES)
    if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.24)
        set(CMAKE_CUDA_ARCHITECTURES native)
    else()
        set(CMAKE_CUDA_ARCHITECTURES 70 80 86)
    endif()
endif()

find_package(MPI REQUIRED COMPONENTS CXX)
find_package(OpenMP REQUIRED COMPONENTS CXX)
find_package(CUDAToolkit REQUIRED)

add_executable(qtclustering qtclustering.cpp)

# The single source file contains CUDA kernels; compile it with nvcc.
set_source_files_properties(qtclustering.cpp PROPERTIES LANGUAGE CUDA)

# -fmad=false (device) and -ffp-contract=off (host) keep GPU and CPU
# distance computations bitwise identical for deterministic results.
target_compile_options(qtclustering PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 -fmad=false -Xcompiler=-O3,-march=native,-fopenmp,-ffp-contract=off,-Wall,-Wextra>)

target_include_directories(qtclustering PRIVATE ${MPI_CXX_INCLUDE_DIRS})

# Link with the host C++ linker so OpenMP/MPI link flags apply directly.
set_target_properties(qtclustering PROPERTIES LINKER_LANGUAGE CXX)

target_link_libraries(qtclustering PRIVATE
    ${MPI_CXX_LIBRARIES}
    OpenMP::OpenMP_CXX
    CUDA::cudart
    m)
