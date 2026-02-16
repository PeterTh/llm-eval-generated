cmake_minimum_required(VERSION 3.10)
project(nbody_benchmark)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(MPI REQUIRED)
add_executable(nbody nbody.cpp)
target_include_directories(nbody PRIVATE ${MPI_CXX_INCLUDE_PATH})
target_link_libraries(nbody PRIVATE ${MPI_CXX_LIBRARIES} m)

# Enable optimizations
target_compile_options(nbody PRIVATE -O3 -march=native -Wall -Wextra)
