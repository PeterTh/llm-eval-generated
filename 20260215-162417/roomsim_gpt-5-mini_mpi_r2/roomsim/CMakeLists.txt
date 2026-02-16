cmake_minimum_required(VERSION 3.10)
project(roomsim_benchmark)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(roomsim roomsim.cpp)

# Enable optimizations and warnings
target_compile_options(roomsim PRIVATE -O3 -march=native -Wall -Wextra -Wpedantic)

# Find and link MPI (unconditionally used)
find_package(MPI REQUIRED)
if(MPI_CXX_FOUND)
    target_link_libraries(roomsim PRIVATE m MPI::MPI_CXX)
else()
    target_link_libraries(roomsim PRIVATE m MPI::MPI_C)
endif()
