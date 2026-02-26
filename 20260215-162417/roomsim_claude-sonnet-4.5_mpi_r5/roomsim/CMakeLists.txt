cmake_minimum_required(VERSION 3.10)
project(roomsim_benchmark)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find MPI
find_package(MPI REQUIRED)

add_executable(roomsim roomsim.cpp)

# Enable optimizations and warnings
target_compile_options(roomsim PRIVATE -O3 -march=native -Wall -Wextra -Wpedantic)

# Include MPI
target_include_directories(roomsim PRIVATE ${MPI_CXX_INCLUDE_DIRS})
target_compile_options(roomsim PRIVATE ${MPI_CXX_COMPILE_FLAGS})

# Link math library and MPI
target_link_libraries(roomsim m ${MPI_CXX_LIBRARIES} ${MPI_CXX_LINK_FLAGS})
