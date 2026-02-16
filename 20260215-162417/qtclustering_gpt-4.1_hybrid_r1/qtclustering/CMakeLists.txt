cmake_minimum_required(VERSION 3.10)

project(qtclustering)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(MPI REQUIRED)
find_package(CUDA REQUIRED)

include_directories(/usr/local/cuda-12.6/targets/x86_64-linux/include)
add_executable(qtclustering qtclustering.cpp)

target_compile_options(qtclustering PRIVATE -O3 -march=native -Wall -Wextra -fopenmp)

# Link math library, MPI, OpenMP, CUDA
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${MPI_CXX_COMPILE_FLAGS} -fopenmp")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${MPI_CXX_LINK_FLAGS}")
target_link_libraries(qtclustering m ${MPI_CXX_LIBRARIES} cuda cudart)
include_directories(${MPI_CXX_INCLUDE_PATH})
