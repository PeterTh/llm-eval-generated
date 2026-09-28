cmake_minimum_required(VERSION 3.18)
project(nbody_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)

# Compile this translation unit as CUDA so it can host both the CUDA kernel and
# the MPI/OpenMP orchestration code in a single file.
set_source_files_properties(nbody.cpp PROPERTIES LANGUAGE CUDA)

add_executable(nbody nbody.cpp)

set_target_properties(nbody PROPERTIES
    CUDA_ARCHITECTURES "70;75;80;86;89;90"
)

target_compile_options(nbody PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 --extended-lambda --fmad=false -Xcompiler=-O3,-Wall,-Wextra,-march=native,-fopenmp>
)

target_include_directories(nbody PRIVATE ${MPI_CXX_INCLUDE_DIRS})
target_link_libraries(nbody PRIVATE MPI::MPI_CXX OpenMP::OpenMP_CXX m)
target_link_options(nbody PRIVATE -fopenmp)
