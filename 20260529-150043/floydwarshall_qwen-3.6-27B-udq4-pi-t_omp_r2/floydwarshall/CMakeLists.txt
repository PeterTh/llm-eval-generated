cmake_minimum_required(VERSION 3.10)
project(floydwarshall_benchmark)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find OpenMP
find_package(OpenMP REQUIRED)

add_executable(floydwarshall floydwarshall.cpp)

# Enable optimizations
if(OPENMP_CXX_FOUND)
    target_compile_options(floydwarshall PRIVATE -O3 -march=native -Wall -Wextra)
    target_link_libraries(floydwarshall PRIVATE OpenMP::OpenMP_CXX m)
else()
    target_compile_options(floydwarshall PRIVATE -O3 -march=native -fopenmp -Wall -Wextra)
    target_link_libraries(floydwarshall PRIVATE -fopenmp m)
endif()
