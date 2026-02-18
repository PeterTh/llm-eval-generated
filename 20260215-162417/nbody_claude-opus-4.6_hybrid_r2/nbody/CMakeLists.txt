cmake_minimum_required(VERSION 3.18)
project(nbody_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)

# Treat the .cpp file as CUDA source
set_source_files_properties(nbody.cpp PROPERTIES LANGUAGE CUDA)

add_executable(nbody nbody.cpp)

# CUDA architecture for RTX 3090 (sm_86), plus common ones
set_target_properties(nbody PROPERTIES
    CUDA_ARCHITECTURES "70;75;80;86;89;90"
)

target_compile_options(nbody PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 --extended-lambda --fmad=false -Xcompiler=-fopenmp,-O3,-march=native,-ffp-contract=off>
)

target_include_directories(nbody PRIVATE ${MPI_CXX_INCLUDE_DIRS})
target_link_libraries(nbody m ${MPI_CXX_LIBRARIES} OpenMP::OpenMP_CXX)
