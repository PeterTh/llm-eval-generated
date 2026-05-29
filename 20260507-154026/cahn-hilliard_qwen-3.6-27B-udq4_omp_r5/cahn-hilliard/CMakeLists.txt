cmake_minimum_required(VERSION 3.10)
project(cahn_hilliard_benchmark)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(cahn_hilliard cahn_hilliard.cpp)

# Enable optimizations and OpenMP
find_package(OpenMP REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")

target_compile_options(cahn_hilliard PRIVATE -O3 -march=native -Wall -Wextra)

# Link math library and OpenMP
target_link_libraries(cahn_hilliard m OpenMP::OpenMP_CXX)
