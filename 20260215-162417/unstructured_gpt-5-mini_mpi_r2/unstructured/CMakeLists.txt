cmake_minimum_required(VERSION 3.10)
project(unstructured_benchmark)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(unstructured unstructured.cpp)

# Enable optimizations
target_compile_options(unstructured PRIVATE -O3 -march=native -Wall -Wextra)

# Link math library and MPI
find_package(MPI REQUIRED)
if(MPI_CXX_FOUND)
  target_link_libraries(unstructured m MPI::MPI_CXX)
  target_compile_definitions(unstructured PRIVATE USE_MPI=1)
else()
  target_link_libraries(unstructured m)
endif()
