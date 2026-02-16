cmake_minimum_required(VERSION 3.10)
project(cholesky_benchmark)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(MPI REQUIRED)

add_executable(cholesky cholesky.cpp)

# Enable optimizations
target_compile_options(cholesky PRIVATE -O3 -march=native -Wall -Wextra)

# Use MPI compilers/flags
target_include_directories(cholesky PRIVATE ${MPI_CXX_INCLUDE_PATH})
target_link_libraries(cholesky PRIVATE m ${MPI_CXX_LIBRARIES})
if(MPI_CXX_LINK_FLAGS)
  set_target_properties(cholesky PROPERTIES LINK_FLAGS "${MPI_CXX_LINK_FLAGS}")
endif()
