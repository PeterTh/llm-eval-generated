cmake_minimum_required(VERSION 3.18)

project(qtclustering LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)

# Try to find CUDA
include(CheckLanguage)
check_language(CUDA)

if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    if(NOT CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES 70 75 80 86 89 90)
    endif()
    find_package(CUDAToolkit REQUIRED)
    set(USE_CUDA ON)
    message(STATUS "CUDA support enabled")
else()
    set(USE_CUDA OFF)
    message(STATUS "CUDA not found - building CPU-only version")
endif()

add_executable(qtclustering qtclustering.cpp)

if(USE_CUDA)
    set_source_files_properties(qtclustering.cpp PROPERTIES LANGUAGE CUDA)
    target_compile_definitions(qtclustering PRIVATE USE_CUDA)
    target_compile_options(qtclustering PRIVATE 
        $<$<COMPILE_LANGUAGE:CUDA>:-O3 --use_fast_math -Xcompiler -fopenmp>
    )
    target_link_libraries(qtclustering CUDA::cudart)
else()
    target_compile_options(qtclustering PRIVATE -O3 -march=native)
endif()

target_include_directories(qtclustering PRIVATE ${MPI_CXX_INCLUDE_DIRS})

# Link libraries
target_link_libraries(qtclustering 
    MPI::MPI_CXX 
    OpenMP::OpenMP_CXX
    m
)
