cmake_minimum_required(VERSION 3.10)

project(qtclustering)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find MPI package
find_package(MPI REQUIRED)

add_executable(qtclustering qtclustering.cpp)

# Include MPI headers and link MPI libraries
target_include_directories(qtclustering PRIVATE ${MPI_CXX_INCLUDE_DIRS})
target_compile_options(qtclustering PRIVATE -O3 -march=native -Wall -Wextra ${MPI_CXX_COMPILE_FLAGS})
target_link_libraries(qtclustering ${MPI_CXX_LIBRARIES} m)

# Add MPI link flags if needed
if(MPI_CXX_LINK_FLAGS)
    set_target_properties(qtclustering PROPERTIES LINK_FLAGS "${MPI_CXX_LINK_FLAGS}")
endif()
