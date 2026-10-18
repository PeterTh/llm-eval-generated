cmake_minimum_required(VERSION 3.18)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES AND CMAKE_VERSION VERSION_GREATER_EQUAL 3.24)
    set(CMAKE_CUDA_ARCHITECTURES native)
endif()

project(qtclustering LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set_source_files_properties(qtclustering.cpp PROPERTIES LANGUAGE CUDA)
add_executable(qtclustering qtclustering.cpp)

target_compile_options(qtclustering PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3;--fmad=false;-Xcompiler=-Wall,-Wextra>
)

# Link math library
target_link_libraries(qtclustering m)
