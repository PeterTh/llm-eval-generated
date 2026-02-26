cmake_minimum_required(VERSION 3.18)

project(qtclustering LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(MPI REQUIRED)

# Compile .cpp as CUDA so nvcc processes it
set_source_files_properties(qtclustering.cpp PROPERTIES LANGUAGE CUDA)

add_executable(qtclustering qtclustering.cpp)

set_target_properties(qtclustering PROPERTIES
    CUDA_ARCHITECTURES "70;75;80;86;89;90"
)

target_compile_options(qtclustering PRIVATE
    -O3 -Xcompiler=-fopenmp,-O3,-march=native,-Wall
)

target_include_directories(qtclustering PRIVATE ${MPI_CXX_INCLUDE_DIRS})
target_link_libraries(qtclustering ${MPI_CXX_LIBRARIES} -fopenmp m)
