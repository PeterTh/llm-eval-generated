cmake_minimum_required(VERSION 3.18)
project(unstructured_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Find MPI and OpenMP
find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)

# Treat .cpp as CUDA source
set_source_files_properties(unstructured.cpp PROPERTIES LANGUAGE CUDA)

add_executable(unstructured unstructured.cpp)

# CUDA architecture (common GPUs)
set_target_properties(unstructured PROPERTIES CUDA_ARCHITECTURES "70;80;90")

# Compile options
target_compile_options(unstructured PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 --extended-lambda -Xcompiler=-fopenmp,-march=native>
)

# Include and link MPI + OpenMP
target_include_directories(unstructured PRIVATE ${MPI_CXX_INCLUDE_DIRS})
target_link_libraries(unstructured ${MPI_CXX_LIBRARIES} OpenMP::OpenMP_CXX m)
target_compile_definitions(unstructured PRIVATE ${MPI_CXX_COMPILE_DEFINITIONS})
