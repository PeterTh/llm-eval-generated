cmake_minimum_required(VERSION 3.10)

project(qtclustering)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(qtclustering qtclustering.cpp)

target_compile_options(qtclustering PRIVATE -O3 -march=native -Wall -Wextra)

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(qtclustering OpenMP::OpenMP_CXX)
endif()
target_link_libraries(qtclustering m)
