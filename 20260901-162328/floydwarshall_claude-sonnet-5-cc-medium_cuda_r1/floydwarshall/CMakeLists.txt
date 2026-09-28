cmake_minimum_required(VERSION 3.18)
project(floydwarshall_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT DEFINED CMAKE_CUDA_STANDARD)
    set(CMAKE_CUDA_STANDARD 20)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 80 86 89)
endif()

find_package(CUDAToolkit REQUIRED)

# The source is CUDA C++; force nvcc to treat the .cpp file as a CUDA
# translation unit so the executable name / file layout stay unchanged.
set_source_files_properties(floydwarshall.cpp PROPERTIES LANGUAGE CUDA)

add_executable(floydwarshall floydwarshall.cpp)

# Enable optimizations
target_compile_options(floydwarshall PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 --use_fast_math>
)

# Link math and CUDA runtime libraries
target_link_libraries(floydwarshall m CUDA::cudart)
