cmake_minimum_required(VERSION 3.10)
project(matmul_benchmark)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(MPI REQUIRED)
add_executable(matmul matmul.cpp)
target_include_directories(matmul PRIVATE ${MPI_CXX_INCLUDE_PATH})
target_compile_options(matmul PRIVATE -O3 -march=native -Wall -Wextra)
target_link_libraries(matmul PRIVATE ${MPI_CXX_LIBRARIES} m)
