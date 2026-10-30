cmake_minimum_required(VERSION 3.10)

project(qtclustering)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)
enable_language(CUDA)
find_package(CUDAToolkit REQUIRED)

add_executable(qtclustering qtclustering.cpp)
set_source_files_properties(qtclustering.cpp PROPERTIES LANGUAGE CUDA)

target_compile_options(qtclustering PRIVATE -O3 -march=native -Wall -Wextra -Xcompiler=-fopenmp)

target_include_directories(qtclustering PRIVATE ${MPI_CXX_INCLUDE_DIRS} ${CUDAToolkit_INCLUDE_DIRS})
target_link_libraries(qtclustering m MPI::MPI_CXX OpenMP::OpenMP_CXX CUDA::cudart)
