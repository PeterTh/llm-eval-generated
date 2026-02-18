cmake_minimum_required(VERSION 3.18)

project(qtclustering LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)

# Treat .cpp as CUDA source
set_source_files_properties(qtclustering.cpp PROPERTIES LANGUAGE CUDA)

add_executable(qtclustering qtclustering.cpp)

# CUDA architecture
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set_target_properties(qtclustering PROPERTIES CUDA_ARCHITECTURES "70;80;90")
endif()

target_compile_options(qtclustering PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 --expt-relaxed-constexpr -Xcompiler=-fopenmp,-march=native>
)

target_include_directories(qtclustering PRIVATE ${MPI_CXX_INCLUDE_DIRS})
target_link_libraries(qtclustering m ${MPI_CXX_LIBRARIES} OpenMP::OpenMP_CXX)
