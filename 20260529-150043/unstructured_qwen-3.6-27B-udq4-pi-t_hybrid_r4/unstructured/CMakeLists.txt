cmake_minimum_required(VERSION 3.10)
project(unstructured CUDA CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 86)

find_package(MPI REQUIRED)

add_executable(unstructured unstructured.cpp)
set_source_files_properties(unstructured.cpp PROPERTIES LANGUAGE CUDA)
target_include_directories(unstructured PRIVATE ${MPI_CXX_INCLUDE_PATH} ${CMAKE_SOURCE_DIR}/..)
target_link_libraries(unstructured ${MPI_CXX_LIBRARIES} ${MPI_CXX_LINK_FLAGS} cuda)
target_compile_options(unstructured PRIVATE -O3 -Xcompiler=-march=native -Xcompiler=-fopenmp)
set_target_properties(unstructured PROPERTIES LINK_FLAGS "-fopenmp")
