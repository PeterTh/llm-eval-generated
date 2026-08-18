cmake_minimum_required(VERSION 3.10)
project(roomsim_benchmark)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(roomsim roomsim.cpp)

# MPI is the required execution model for this benchmark.  Link through the
# imported target so the MPI compiler wrapper and all platform-specific flags
# are selected by CMake.
find_package(MPI REQUIRED COMPONENTS CXX)

# Enable optimizations and warnings
target_compile_options(roomsim PRIVATE -O3 -march=native -Wall -Wextra -Wpedantic)

# Link math library
target_link_libraries(roomsim PRIVATE MPI::MPI_CXX m)
