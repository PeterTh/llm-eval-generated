cmake_minimum_required(VERSION 3.10)

project(qtclustering)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(qtclustering qtclustering.cpp)

target_compile_options(qtclustering PRIVATE -O3 -march=native -Wall -Wextra)

# Link math and MPI libraries
find_package(MPI REQUIRED)
target_include_directories(qtclustering PRIVATE ${MPI_CXX_INCLUDE_PATH})
target_link_libraries(qtclustering PRIVATE m MPI::MPI_CXX)
