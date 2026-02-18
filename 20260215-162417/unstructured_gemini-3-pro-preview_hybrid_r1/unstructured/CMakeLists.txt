cmake_minimum_required(VERSION 3.18)
project(unstructured_benchmark LANGUAGES CXX CUDA)

find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)

add_executable(unstructured unstructured.cpp)

set_source_files_properties(unstructured.cpp PROPERTIES LANGUAGE CUDA)

# Enable optimizations
target_compile_options(unstructured PRIVATE 
    $<$<COMPILE_LANGUAGE:CXX>:-O3 -march=native -Wall -Wextra>
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 -Xcompiler -Wall,-Wextra>
)

# Link libraries
target_link_libraries(unstructured PRIVATE m MPI::MPI_CXX OpenMP::OpenMP_CXX)
