cmake_minimum_required(VERSION 3.10)
project(roomsim_benchmark LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)

add_executable(roomsim roomsim.cpp)

target_include_directories(roomsim PRIVATE ${MPI_CXX_INCLUDE_DIRS})
target_compile_options(roomsim PRIVATE -O3 -march=native -Wall -Wextra -Wpedantic ${MPI_CXX_COMPILE_FLAGS} ${OpenMP_CXX_FLAGS})
target_link_libraries(roomsim PRIVATE ${MPI_CXX_LIBRARIES} OpenMP::OpenMP_CXX m)
