cmake_minimum_required(VERSION 3.10)

project(qtclustering)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find MPI package
find_package(MPI REQUIRED)

add_executable(qtclustering qtclustering.cpp)

# Add MPI include directories and compile definitions
target_include_directories(qtclustering PRIVATE ${MPI_CXX_INCLUDE_DIRS})
target_compile_definitions(qtclustering PRIVATE ${MPI_CXX_COMPILE_DEFINITIONS})

target_compile_options(qtclustering PRIVATE -O3 -march=native -Wall -Wextra ${MPI_CXX_COMPILE_OPTIONS})

# Link MPI and math libraries
target_link_libraries(qtclustering ${MPI_CXX_LIBRARIES} m)
target_link_options(qtclustering PRIVATE ${MPI_CXX_LINK_FLAGS})
