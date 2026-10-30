cmake_minimum_required(VERSION 3.10)
project(floydwarshall_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)

find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)

add_executable(floydwarshall floydwarshall.cpp)
set_source_files_properties(floydwarshall.cpp PROPERTIES LANGUAGE CUDA)
set_target_properties(floydwarshall PROPERTIES CUDA_ARCHITECTURES "70;75;80;86;89;90")

target_include_directories(floydwarshall PRIVATE ${MPI_CXX_INCLUDE_DIRS})
target_link_libraries(floydwarshall PRIVATE MPI::MPI_CXX OpenMP::OpenMP_CXX ${CUDA_LIBRARIES} m)

target_compile_options(floydwarshall PRIVATE -O3 -Xcompiler=-fopenmp -Xcompiler=-march=native)
