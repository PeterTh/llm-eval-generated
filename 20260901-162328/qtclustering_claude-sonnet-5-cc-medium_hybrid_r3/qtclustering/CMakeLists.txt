cmake_minimum_required(VERSION 3.18)

project(qtclustering LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES OR CMAKE_CUDA_ARCHITECTURES STREQUAL "")
    set(CMAKE_CUDA_ARCHITECTURES native)
endif()

# Compile the single source file as CUDA so it can host both the CUDA
# kernels and the MPI/OpenMP driven host code in one translation unit.
set_source_files_properties(qtclustering.cpp PROPERTIES LANGUAGE CUDA)

add_executable(qtclustering qtclustering.cpp)

target_include_directories(qtclustering PRIVATE ${MPI_CXX_INCLUDE_DIRS})

target_compile_options(qtclustering PRIVATE
    -O3
    --extended-lambda
    -Xcompiler=-O3,-Wall,-Wextra,-fopenmp
)

target_link_options(qtclustering PRIVATE -fopenmp)

target_link_libraries(qtclustering PRIVATE ${MPI_CXX_LIBRARIES} OpenMP::OpenMP_CXX m)
