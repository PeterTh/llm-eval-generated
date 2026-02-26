cmake_minimum_required(VERSION 3.18)
project(unstructured_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)

set_source_files_properties(unstructured.cpp PROPERTIES LANGUAGE CUDA)
add_executable(unstructured unstructured.cpp)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set_target_properties(unstructured PROPERTIES CUDA_ARCHITECTURES "70;80;90")
endif()

target_include_directories(unstructured PRIVATE ${MPI_CXX_INCLUDE_DIRS})
target_compile_options(unstructured PRIVATE
    -O3
    -Xcompiler=-fopenmp
    -Xcompiler=-march=native
    -Xcompiler=-Wall
    -Xcompiler=-Wextra
)
target_link_libraries(unstructured ${MPI_CXX_LIBRARIES} ${OpenMP_CXX_LIBRARIES} m)
