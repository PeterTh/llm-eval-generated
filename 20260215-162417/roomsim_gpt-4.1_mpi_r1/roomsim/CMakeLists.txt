cmake_minimum_required(VERSION 3.10)
project(roomsim_benchmark)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(MPI REQUIRED)
add_executable(roomsim roomsim.cpp)
target_include_directories(roomsim PRIVATE ${MPI_CXX_INCLUDE_PATH})
target_link_libraries(roomsim PRIVATE ${MPI_CXX_LIBRARIES} m)
target_compile_options(roomsim PRIVATE ${MPI_CXX_COMPILE_FLAGS})
# Remove previous target_link_libraries for m


# Enable optimizations and warnings
target_compile_options(roomsim PRIVATE -O3 -march=native -Wall -Wextra -Wpedantic)

# Link math library
