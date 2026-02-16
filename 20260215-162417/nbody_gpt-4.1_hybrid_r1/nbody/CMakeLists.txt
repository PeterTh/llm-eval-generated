cmake_minimum_required(VERSION 3.10)
project(nbody_benchmark)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)

enable_language(CUDA)

add_executable(nbody nbody.cpp)
set_target_properties(nbody PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(nbody PROPERTIES LINKER_LANGUAGE CUDA)

# Enable optimizations
target_compile_options(nbody PRIVATE -O3 -march=native -Wall -Wextra)

# Link math library and parallel libraries
target_link_libraries(nbody m MPI::MPI_CXX OpenMP::OpenMP_CXX cuda)

target_compile_definitions(nbody PRIVATE _USE_MATH_DEFINES)
