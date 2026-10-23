cmake_minimum_required(VERSION 3.21)
project(cahn_hilliard_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES OR CMAKE_CUDA_ARCHITECTURES STREQUAL "52")
    if(CMAKE_VERSION VERSION_GREATER_EQUAL "3.24")
        set(CMAKE_CUDA_ARCHITECTURES native CACHE STRING "CUDA architectures" FORCE)
    else()
        set(CMAKE_CUDA_ARCHITECTURES "70;75;80;86" CACHE STRING "CUDA architectures" FORCE)
    endif()
endif()

find_package(MPI REQUIRED COMPONENTS CXX)
find_package(OpenMP REQUIRED)
find_package(CUDAToolkit REQUIRED)

add_executable(cahn_hilliard cahn_hilliard.cpp)
set_source_files_properties(cahn_hilliard.cpp PROPERTIES LANGUAGE CUDA)

target_include_directories(cahn_hilliard PRIVATE ${MPI_CXX_INCLUDE_DIRS})

set(CUDA_HOST_OPTIONS "-O3,-march=native,${OpenMP_CXX_FLAGS}")
target_compile_options(cahn_hilliard PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3;--fmad=false;-Xcompiler=${CUDA_HOST_OPTIONS}>
)

target_link_libraries(cahn_hilliard PRIVATE
    ${MPI_CXX_LIBRARIES}
    ${OpenMP_CXX_LIBRARIES}
    CUDA::cudart
    m
)

target_link_options(cahn_hilliard PRIVATE
    $<$<LINK_LANGUAGE:CUDA>:${OpenMP_CXX_FLAGS}>
)
