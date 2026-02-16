cmake_minimum_required(VERSION 3.10)
project(unstructured_benchmark LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(unstructured unstructured.cpp)

# Enable optimizations
target_compile_options(unstructured PRIVATE -O3 -march=native -Wall -Wextra)

# OpenMP
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_compile_definitions(unstructured PRIVATE _OPENMP)
    target_link_libraries(unstructured OpenMP::OpenMP_CXX)
endif()

# MPI
find_package(MPI)
if(MPI_CXX_FOUND)
    target_compile_definitions(unstructured PRIVATE USE_MPI)
    target_include_directories(unstructured PRIVATE ${MPI_CXX_INCLUDE_PATH})
    target_link_libraries(unstructured ${MPI_CXX_LIBRARIES})
endif()

# Link math library
target_link_libraries(unstructured m)
