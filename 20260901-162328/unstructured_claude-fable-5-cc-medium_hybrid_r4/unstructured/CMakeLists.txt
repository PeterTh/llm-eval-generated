cmake_minimum_required(VERSION 3.18)
project(unstructured_benchmark LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# nvcc requires a supported host compiler (CUDA 12.0 supports up to GCC 12)
if(NOT CMAKE_CUDA_HOST_COMPILER AND NOT DEFINED ENV{CUDAHOSTCXX})
    find_program(GXX12 g++-12)
    if(GXX12)
        set(CMAKE_CUDA_HOST_COMPILER ${GXX12})
    endif()
endif()
enable_language(CUDA)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
if(NOT CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES native)
endif()

find_package(MPI REQUIRED COMPONENTS CXX)
find_package(OpenMP REQUIRED COMPONENTS CXX)

add_executable(unstructured unstructured.cpp)
set_source_files_properties(unstructured.cpp PROPERTIES LANGUAGE CUDA)

# Enable optimizations; forward host-side flags (OpenMP, arch tuning) via nvcc
target_compile_options(unstructured PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 -Xcompiler=-O3,-march=native,-fopenmp,-Wall,-Wextra>)

target_include_directories(unstructured PRIVATE ${MPI_CXX_INCLUDE_DIRS})
target_link_libraries(unstructured PRIVATE ${MPI_CXX_LIBRARIES} m)
target_link_options(unstructured PRIVATE -fopenmp)
