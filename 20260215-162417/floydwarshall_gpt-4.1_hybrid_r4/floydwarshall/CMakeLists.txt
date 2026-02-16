cmake_minimum_required(VERSION 3.10)
project(floydwarshall_benchmark)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)
find_package(CUDA)

add_executable(floydwarshall floydwarshall.cpp)

# Enable optimizations
target_compile_options(floydwarshall PRIVATE -O3 -march=native -Wall -Wextra -fopenmp)

# Link libraries
if(CUDA_FOUND)
    target_link_libraries(floydwarshall PUBLIC m MPI::MPI_CXX ${CUDA_LIBRARIES} OpenMP::OpenMP_CXX)
    target_include_directories(floydwarshall PUBLIC ${CUDA_INCLUDE_DIRS})
else()
    target_link_libraries(floydwarshall PUBLIC m MPI::MPI_CXX OpenMP::OpenMP_CXX)
endif()
