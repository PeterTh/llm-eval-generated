cmake_minimum_required(VERSION 3.10)
project(unstructured_benchmark LANGUAGES CXX CUDA)

find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)

add_executable(unstructured unstructured.cpp)
set_source_files_properties(unstructured.cpp PROPERTIES LANGUAGE CUDA)

target_compile_options(unstructured PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:-O3>
)

target_link_libraries(unstructured PRIVATE MPI::MPI_CXX OpenMP::OpenMP_CXX cudart m)
