cmake_minimum_required(VERSION 3.18)
project(floydwarshall_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(floydwarshall floydwarshall.cpp)
set_source_files_properties(floydwarshall.cpp PROPERTIES LANGUAGE CUDA)
set_target_properties(floydwarshall PROPERTIES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON)
if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.24)
    set_property(TARGET floydwarshall PROPERTY CUDA_ARCHITECTURES native)
endif()

# Enable optimizations
target_compile_options(floydwarshall PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3>)

# Link math library
target_link_libraries(floydwarshall m)
