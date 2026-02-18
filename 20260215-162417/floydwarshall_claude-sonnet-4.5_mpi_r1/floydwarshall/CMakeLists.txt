cmake_minimum_required(VERSION 3.10)
project(floydwarshall_benchmark)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find MPI
find_package(MPI REQUIRED)

add_executable(floydwarshall floydwarshall.cpp)

# Enable optimizations
target_compile_options(floydwarshall PRIVATE -O3 -march=native -Wall -Wextra)

# Include MPI headers and link MPI libraries
target_include_directories(floydwarshall PRIVATE ${MPI_CXX_INCLUDE_DIRS})
target_link_libraries(floydwarshall ${MPI_CXX_LIBRARIES} m)
target_compile_options(floydwarshall PRIVATE ${MPI_CXX_COMPILE_FLAGS})
target_link_options(floydwarshall PRIVATE ${MPI_CXX_LINK_FLAGS})
