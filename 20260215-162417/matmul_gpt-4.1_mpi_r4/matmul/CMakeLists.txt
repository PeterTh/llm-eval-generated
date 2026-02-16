cmake_minimum_required(VERSION 3.10)
project(matmul_benchmark)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(matmul matmul.cpp)

# Enable optimizations
target_compile_options(matmul PRIVATE -O3 -march=native -Wall -Wextra)

# Find and link MPI
find_package(MPI REQUIRED)
target_include_directories(matmul PRIVATE ${MPI_CXX_INCLUDE_PATH})
target_link_libraries(matmul PRIVATE ${MPI_CXX_LIBRARIES} m)
set_target_properties(matmul PROPERTIES COMPILE_FLAGS "${MPI_CXX_COMPILE_FLAGS}" LINK_FLAGS "${MPI_CXX_LINK_FLAGS}")
