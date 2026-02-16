cmake_minimum_required(VERSION 3.10)
project(floydwarshall_benchmark)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)
enable_language(CUDA)

add_executable(floydwarshall floydwarshall.cpp)

# Enable optimizations
target_compile_options(floydwarshall PRIVATE -O3 -march=native -Wall -Wextra $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler -O3>)

# Link math library and parallel libraries
target_link_libraries(floydwarshall m MPI::MPI_CXX OpenMP::OpenMP_CXX cuda)
set_target_properties(floydwarshall PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
