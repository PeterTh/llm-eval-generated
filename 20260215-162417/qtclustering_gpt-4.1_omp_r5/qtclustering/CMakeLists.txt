cmake_minimum_required(VERSION 3.10)

project(qtclustering)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(qtclustering qtclustering.cpp)

find_package(OpenMP REQUIRED)
target_compile_options(qtclustering PRIVATE -O3 -march=native -Wall -Wextra $<$<BOOL:${OpenMP_CXX_FOUND}>:${OpenMP_CXX_FLAGS}>)

# Link math library and OpenMP
target_link_libraries(qtclustering m $<$<BOOL:${OpenMP_CXX_FOUND}>:${OpenMP_CXX_LIBRARIES}>)
