cmake_minimum_required(VERSION 3.10)

project(qtclustering LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find required packages
find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)

# Add subdirectory for the executable
add_subdirectory(qtclustering)
