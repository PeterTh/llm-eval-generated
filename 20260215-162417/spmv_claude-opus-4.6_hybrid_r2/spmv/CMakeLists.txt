cmake_minimum_required(VERSION 3.18)
project(spmv_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)

# Compile spmv.cpp as CUDA
set_source_files_properties(spmv.cpp PROPERTIES LANGUAGE CUDA)

add_executable(spmv spmv.cpp)

set_target_properties(spmv PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_compile_options(spmv PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 --extended-lambda -Xcompiler=-fopenmp,-O3,-march=native>
)

target_include_directories(spmv PRIVATE ${MPI_CXX_INCLUDE_DIRS})
target_link_libraries(spmv m ${MPI_CXX_LIBRARIES} OpenMP::OpenMP_CXX)
