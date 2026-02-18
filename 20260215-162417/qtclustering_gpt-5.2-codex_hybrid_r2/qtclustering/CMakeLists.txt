cmake_minimum_required(VERSION 3.10)

if (NOT DEFINED CMAKE_CUDA_ARCHITECTURES OR CMAKE_CUDA_ARCHITECTURES STREQUAL "")
    set(CMAKE_CUDA_ARCHITECTURES 70 CACHE STRING "CUDA architectures" FORCE)
endif()

project(qtclustering LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)

add_executable(qtclustering qtclustering.cpp)
set_source_files_properties(qtclustering.cpp PROPERTIES LANGUAGE CUDA)

target_compile_options(qtclustering PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-O3 -march=native -Wall -Wextra>
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 -Xcompiler=-fopenmp>
)

target_link_libraries(qtclustering PRIVATE
    MPI::MPI_CXX
    OpenMP::OpenMP_CXX
    m
)

if (TARGET OpenMP::OpenMP_CUDA)
    target_link_libraries(qtclustering PRIVATE OpenMP::OpenMP_CUDA)
endif()
