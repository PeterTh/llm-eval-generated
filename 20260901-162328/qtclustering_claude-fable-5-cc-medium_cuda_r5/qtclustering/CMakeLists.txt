cmake_minimum_required(VERSION 3.18)

project(qtclustering LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# The source is CUDA C++ (kept under its original name)
set_source_files_properties(qtclustering.cpp PROPERTIES LANGUAGE CUDA)

add_executable(qtclustering qtclustering.cpp)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES OR CMAKE_CUDA_ARCHITECTURES STREQUAL "")
    set_target_properties(qtclustering PROPERTIES CUDA_ARCHITECTURES "86")
endif()

target_compile_options(qtclustering PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 --fmad=false -Xcompiler=-O3,-Wall,-Wextra>)

# Link math library
target_link_libraries(qtclustering m)
