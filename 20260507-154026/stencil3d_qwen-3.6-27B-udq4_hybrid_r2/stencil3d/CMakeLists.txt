cmake_minimum_required(VERSION 3.18)
project(stencil3d_benchmark)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

enable_language(CUDA)
set(CMAKE_CUDA_ARCHITECTURES 86)

find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)

add_executable(stencil3d stencil3d.cpp)
set_source_files_properties(stencil3d.cpp PROPERTIES LANGUAGE CUDA)

target_include_directories(stencil3d PRIVATE ${MPI_CXX_INCLUDE_PATH})
target_link_libraries(stencil3d MPI::MPI_CXX ${OpenMP_CXX_LIBRARIES} m)

target_compile_options(stencil3d PRIVATE -O3 -Xcompiler -march=native,-Wall,-Wextra,-fopenmp)
