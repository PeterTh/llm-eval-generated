cmake_minimum_required(VERSION 3.18)
project(nbody_benchmark)

enable_language(CUDA)
find_package(MPI REQUIRED COMPONENTS CXX)
find_package(OpenMP REQUIRED COMPONENTS CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(nbody nbody.cpp)
set_source_files_properties(nbody.cpp PROPERTIES LANGUAGE CUDA)
set_target_properties(nbody PROPERTIES CUDA_STANDARD 20 CUDA_STANDARD_REQUIRED ON)

target_compile_options(nbody PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3;--use_fast_math> $<$<COMPILE_LANGUAGE:CXX>:-O3;-march=native;-Wall;-Wextra>)
target_link_libraries(nbody PRIVATE MPI::MPI_CXX OpenMP::OpenMP_CXX m)
