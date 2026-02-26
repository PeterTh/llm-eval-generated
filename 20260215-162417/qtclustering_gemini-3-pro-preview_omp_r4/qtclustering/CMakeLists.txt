cmake_minimum_required(VERSION 3.10)

project(qtclustering)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenMP)

add_executable(qtclustering qtclustering.cpp)

if(OpenMP_CXX_FOUND)
    target_link_libraries(qtclustering PRIVATE OpenMP::OpenMP_CXX)
endif()

target_compile_options(qtclustering PRIVATE -O3 -march=native -Wall -Wextra)

# Link math library
target_link_libraries(qtclustering PRIVATE m)
