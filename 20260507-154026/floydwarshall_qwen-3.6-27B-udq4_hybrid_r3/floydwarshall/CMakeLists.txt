cmake_minimum_required(VERSION 3.10)
project(floydwarshall_benchmark)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

enable_language(CUDA)
find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)

add_executable(floydwarshall floydwarshall.cpp)
set_source_files_properties(floydwarshall.cpp PROPERTIES LANGUAGE CUDA)

# Host + device optimizations
target_compile_options(floydwarshall PRIVATE -O3 -Xcompiler=-march=native -Wall -Wextra)

# Link MPI, OpenMP, and math
target_link_libraries(floydwarshall MPI::MPI_CXX OpenMP::OpenMP_CXX m)
