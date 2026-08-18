cmake_minimum_required(VERSION 3.24)

if(DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(QTCLUSTERING_USER_CUDA_ARCHITECTURES TRUE)
else()
    set(QTCLUSTERING_USER_CUDA_ARCHITECTURES FALSE)
endif()

project(qtclustering LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(MPI REQUIRED COMPONENTS CXX)
find_package(OpenMP REQUIRED COMPONENTS CXX)
find_package(CUDAToolkit REQUIRED)

add_executable(qtclustering qtclustering.cpp)
set_source_files_properties(qtclustering.cpp PROPERTIES LANGUAGE CUDA)

# Produce portable accelerator-cluster fat binaries unless the caller supplies
# the exact architecture(s) for its deployment GPUs.
if(NOT QTCLUSTERING_USER_CUDA_ARCHITECTURES)
    if(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 13.0)
        set_property(TARGET qtclustering PROPERTY CUDA_ARCHITECTURES 75 80 86 89 90)
    elseif(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 12.0)
        set_property(TARGET qtclustering PROPERTY CUDA_ARCHITECTURES 60 70 75 80 86 89 90)
    elseif(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 11.8)
        set_property(TARGET qtclustering PROPERTY CUDA_ARCHITECTURES 60 70 75 80 86 89)
    else()
        set_property(TARGET qtclustering PROPERTY CUDA_ARCHITECTURES 60 70 75 80)
    endif()
endif()

target_compile_options(qtclustering PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3;--expt-relaxed-constexpr;-Xcompiler=-O3,-march=native,-Wall,-Wextra,-fopenmp>
    $<$<COMPILE_LANGUAGE:CXX>:-O3;-march=native;-Wall;-Wextra>
)

target_link_libraries(qtclustering PRIVATE
    MPI::MPI_CXX
    OpenMP::OpenMP_CXX
    CUDA::cudart
    m
)
