cmake_minimum_required(VERSION 3.10)

project(qtclustering)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

enable_language(CUDA)
add_executable(qtclustering qtclustering.cpp)
target_sources(qtclustering PRIVATE qtclustering.cpp)
target_compile_options(qtclustering PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3 -arch=sm_60>)

target_compile_options(qtclustering PRIVATE -O3 -march=native -Wall -Wextra $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall,-Wextra>)

# Link math library
target_link_libraries(qtclustering m)
