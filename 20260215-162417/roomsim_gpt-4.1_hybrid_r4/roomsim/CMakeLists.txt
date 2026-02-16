cmake_minimum_required(VERSION 3.10)
project(roomsim_benchmark)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(MPI REQUIRED)

add_executable(roomsim roomsim.cpp)

# Enable optimizations and warnings
target_compile_options(roomsim PRIVATE -O3 -march=native -Wall -Wextra -Wpedantic -fopenmp)

target_include_directories(roomsim PRIVATE ${MPI_INCLUDE_PATH})
target_link_libraries(roomsim PRIVATE ${MPI_LIBRARIES} m)
target_link_options(roomsim PRIVATE -fopenmp)
