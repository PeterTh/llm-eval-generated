cmake_minimum_required(VERSION 3.10)
project(nbody_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)

find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)

add_executable(nbody nbody.cpp)
set_source_files_properties(nbody.cpp PROPERTIES LANGUAGE CUDA)

target_include_directories(nbody PRIVATE ${MPI_CXX_INCLUDE_DIRS})
target_compile_options(nbody PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 --use_fast_math>
    $<$<COMPILE_LANGUAGE:CXX>:-O3 -march=native -Wall -Wextra>
)
target_link_libraries(nbody PRIVATE MPI::MPI_CXX OpenMP::OpenMP_CXX m cudart)
