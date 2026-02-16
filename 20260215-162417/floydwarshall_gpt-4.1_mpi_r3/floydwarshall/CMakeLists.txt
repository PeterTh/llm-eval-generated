cmake_minimum_required(VERSION 3.10)
project(floydwarshall_benchmark)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(floydwarshall floydwarshall.cpp)

# Enable optimizations
target_compile_options(floydwarshall PRIVATE -O3 -march=native -Wall -Wextra)

# Find and link MPI
find_package(MPI REQUIRED)
target_include_directories(floydwarshall PRIVATE ${MPI_CXX_INCLUDE_PATH})
target_link_libraries(floydwarshall PRIVATE ${MPI_CXX_LIBRARIES} m)
