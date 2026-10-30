cmake_minimum_required(VERSION 3.10)
project(spmv_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 14)

find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)

add_executable(spmv spmv.cpp)
set_source_files_properties(spmv.cpp PROPERTIES LANGUAGE CUDA)

target_compile_options(spmv PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:-O3>
  $<$<COMPILE_LANGUAGE:CXX>:-O3 -march=native -Wall -Wextra>
)
target_compile_options(spmv PRIVATE -Xcompiler -fopenmp)

target_include_directories(spmv PRIVATE ${MPI_CXX_INCLUDE_DIRS})
target_link_libraries(spmv PRIVATE MPI::MPI_CXX OpenMP::OpenMP_CXX ${CMAKE_CUDA_RUNTIME_LIBRARY} m)
