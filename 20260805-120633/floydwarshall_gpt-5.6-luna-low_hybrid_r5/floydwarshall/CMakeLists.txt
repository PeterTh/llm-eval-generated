cmake_minimum_required(VERSION 3.10)
project(floydwarshall_benchmark)

enable_language(CUDA)
find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(floydwarshall floydwarshall.cpp)
set_source_files_properties(floydwarshall.cpp PROPERTIES LANGUAGE CUDA)

# Enable optimizations
target_compile_options(floydwarshall PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-O3;-march=native;-Wall;-Wextra>
                                      $<$<COMPILE_LANGUAGE:CUDA>:-O3;-Xcompiler=-fopenmp>)
set_target_properties(floydwarshall PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# Link math library
target_link_libraries(floydwarshall PRIVATE MPI::MPI_CXX OpenMP::OpenMP_CXX m)
