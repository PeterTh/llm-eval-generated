cmake_minimum_required(VERSION 3.10)
project(roomsim_benchmark)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(roomsim roomsim.cpp)
find_package(OpenMP REQUIRED)
target_link_libraries(roomsim m OpenMP::OpenMP_CXX)

# Enable optimizations and warnings
target_compile_options(roomsim PRIVATE -O3 -march=native -Wall -Wextra -Wpedantic $<$<COMPILE_LANG_AND_ID:CXX,GNU>:-fopenmp> $<$<COMPILE_LANG_AND_ID:CXX,Clang>:-fopenmp>)

# Link math library
