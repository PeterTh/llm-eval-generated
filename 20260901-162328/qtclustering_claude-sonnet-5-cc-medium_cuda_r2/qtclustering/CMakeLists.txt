cmake_minimum_required(VERSION 3.18)

project(qtclustering LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES native)
endif()

add_executable(qtclustering qtclustering.cpp)

# Compile the source as CUDA even though it keeps the .cpp extension.
set_source_files_properties(qtclustering.cpp PROPERTIES LANGUAGE CUDA)

set_target_properties(qtclustering PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

target_compile_options(qtclustering PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 --use_fast_math>
)

target_link_libraries(qtclustering m)
