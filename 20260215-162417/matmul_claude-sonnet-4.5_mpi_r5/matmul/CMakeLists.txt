cmake_minimum_required(VERSION 3.10)
project(matmul_benchmark)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find MPI
find_package(MPI REQUIRED)

add_executable(matmul matmul.cpp)

# Enable optimizations
target_compile_options(matmul PRIVATE -O3 -march=native -Wall -Wextra)

# Include MPI headers and link MPI libraries
target_include_directories(matmul PRIVATE ${MPI_CXX_INCLUDE_DIRS})
target_link_libraries(matmul m ${MPI_CXX_LIBRARIES})

# Add MPI compile flags
if(MPI_CXX_COMPILE_FLAGS)
    set_target_properties(matmul PROPERTIES COMPILE_FLAGS "${MPI_CXX_COMPILE_FLAGS}")
endif()

# Add MPI link flags
if(MPI_CXX_LINK_FLAGS)
    set_target_properties(matmul PROPERTIES LINK_FLAGS "${MPI_CXX_LINK_FLAGS}")
endif()
