cmake_minimum_required(VERSION 3.10)
project(roomsim_benchmark LANGUAGES CXX)

find_package(MPI REQUIRED)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(roomsim roomsim.cpp)

target_compile_options(roomsim PRIVATE -O3 -march=native -Wall -Wextra -Wpedantic)

target_link_libraries(roomsim m MPI::MPI_CXX)
