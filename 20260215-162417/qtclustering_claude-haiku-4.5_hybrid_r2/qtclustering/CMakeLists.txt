cmake_minimum_required(VERSION 3.10)

project(qtclustering LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find required packages
find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)

add_executable(qtclustering qtclustering.cpp)

# Compile options
target_compile_options(qtclustering PRIVATE 
    $<$<COMPILE_LANGUAGE:CXX>:-O3 -march=native -Wall -Wextra>
)

# Link libraries
target_link_libraries(qtclustering 
    PRIVATE m
    PRIVATE MPI::MPI_CXX
    PRIVATE OpenMP::OpenMP_CXX
)

# Include directories
target_include_directories(qtclustering 
    PRIVATE ${MPI_CXX_INCLUDE_DIRS}
)
