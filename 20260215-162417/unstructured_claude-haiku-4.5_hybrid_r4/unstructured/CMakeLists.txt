cmake_minimum_required(VERSION 3.10)
project(unstructured_benchmark)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find MPI
find_package(MPI REQUIRED)

# Find OpenMP
find_package(OpenMP REQUIRED)

add_executable(unstructured unstructured.cpp)

# Enable optimizations
target_compile_options(unstructured PRIVATE -O3 -march=native -Wall -Wextra)

# Link MPI, OpenMP, and math library
target_link_libraries(unstructured MPI::MPI_CXX OpenMP::OpenMP_CXX m)

# Set MPI compiler wrapper if available
if(MPI_CXX_FOUND)
    target_include_directories(unstructured PRIVATE ${MPI_CXX_INCLUDE_DIRS})
endif()

