cmake_minimum_required(VERSION 3.10)
project(floydwarshall_benchmark)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenMP REQUIRED)

add_executable(floydwarshall floydwarshall.cpp)

target_link_libraries(floydwarshall PUBLIC OpenMP::OpenMP_CXX m)
