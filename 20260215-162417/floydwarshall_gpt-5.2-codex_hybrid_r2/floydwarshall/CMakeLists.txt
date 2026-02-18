cmake_minimum_required(VERSION 3.10)
if (NOT DEFINED CMAKE_CUDA_ARCHITECTURES OR CMAKE_CUDA_ARCHITECTURES STREQUAL "")
    set(CMAKE_CUDA_ARCHITECTURES 70 80 90 CACHE STRING "" FORCE)
endif()
project(floydwarshall_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

add_executable(floydwarshall floydwarshall.cpp)
set_source_files_properties(floydwarshall.cpp PROPERTIES LANGUAGE CUDA)

find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)

# Enable optimizations
target_compile_options(floydwarshall PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-O3 -march=native -Wall -Wextra>
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 -Xcompiler=-fopenmp -Xcompiler=-march=native>
)

# Link math library
target_link_libraries(floydwarshall PRIVATE m MPI::MPI_CXX OpenMP::OpenMP_CXX)
