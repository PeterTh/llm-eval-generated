cmake_minimum_required(VERSION 3.10)

# Set CUDA architecture before project() to enable proper detection
set(CMAKE_CUDA_ARCHITECTURES 60 70 80)

project(qtclustering CUDA CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(CUDA REQUIRED)

add_executable(qtclustering qtclustering.cpp)

# Set source file properties to compile as CUDA
set_source_files_properties(qtclustering.cpp PROPERTIES LANGUAGE CUDA)

set_target_properties(qtclustering PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    LANGUAGE CUDA
)

target_include_directories(qtclustering PRIVATE ${CUDA_INCLUDE_DIRS})

target_compile_options(qtclustering PRIVATE 
    $<$<COMPILE_LANGUAGE:CXX>:-O3 -march=native -Wall -Wextra>
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 --compiler-options=-march=native>
)

target_link_libraries(qtclustering m ${CUDA_LIBRARIES})
