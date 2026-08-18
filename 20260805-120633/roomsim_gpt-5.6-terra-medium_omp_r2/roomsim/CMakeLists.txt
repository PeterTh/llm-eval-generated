cmake_minimum_required(VERSION 3.10)
project(roomsim_benchmark)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(roomsim roomsim.cpp)

# Enable optimizations and warnings
target_compile_options(roomsim PRIVATE -O3 -march=native -Wall -Wextra -Wpedantic)

# OpenMP is required: the benchmark always executes its compute phases in the
# shared-memory parallel implementation.
find_package(OpenMP REQUIRED)

# Link math library and OpenMP runtime
target_link_libraries(roomsim PRIVATE m OpenMP::OpenMP_CXX)
