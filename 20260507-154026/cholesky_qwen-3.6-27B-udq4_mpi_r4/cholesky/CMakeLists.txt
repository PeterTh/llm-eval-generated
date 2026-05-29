cmake_minimum_required(VERSION 3.10)
project(cholesky_benchmark C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(MPI REQUIRED)

add_executable(cholesky cholesky.cpp)

# Enable optimizations
target_compile_options(cholesky PRIVATE -O3 -march=native -Wall -Wextra)

# Link MPI and math library
target_include_directories(cholesky PRIVATE ${MPI_CXX_INCLUDE_PATH} ../common)
target_link_libraries(cholesky ${MPI_CXX_LIBRARIES} m)
