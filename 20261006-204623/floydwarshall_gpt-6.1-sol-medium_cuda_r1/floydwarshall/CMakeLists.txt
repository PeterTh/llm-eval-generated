cmake_minimum_required(VERSION 3.24)
# Optimize for the installed GPU unless the caller selects target architectures.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES native)
endif()
project(floydwarshall_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Compile the existing source as CUDA without changing its filename.
set_source_files_properties(floydwarshall.cpp PROPERTIES LANGUAGE CUDA)
add_executable(floydwarshall floydwarshall.cpp)
target_compile_options(floydwarshall PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3;-Xcompiler=-Wall,-Wextra>)
target_link_libraries(floydwarshall PRIVATE m)
