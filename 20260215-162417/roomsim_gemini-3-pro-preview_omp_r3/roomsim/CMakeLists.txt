cmake_minimum_required(VERSION 3.10)
project(roomsim_benchmark)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(roomsim roomsim.cpp)

# Enable optimizations and warnings
target_compile_options(roomsim PRIVATE -O3 -march=native -Wall -Wextra -Wpedantic)

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(roomsim OpenMP::OpenMP_CXX)
endif()

# Link math library
target_link_libraries(roomsim m)
