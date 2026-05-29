cmake_minimum_required(VERSION 3.10)
project(floydwarshall_benchmark)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(floydwarshall floydwarshall.cpp)

# Enable optimizations with OpenMP
find_package(OpenMP REQUIRED)
target_compile_options(floydwarshall PRIVATE -O3 -march=native -Wall -Wextra ${OpenMP_CXX_FLAGS})
target_link_libraries(floydwarshall ${OpenMP_CXX_FLAGS} m)
