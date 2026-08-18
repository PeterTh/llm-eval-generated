cmake_minimum_required(VERSION 3.10)
project(nbody_benchmark LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(OpenMP REQUIRED COMPONENTS CXX)

add_executable(nbody nbody.cpp)

# Enable optimizations and require OpenMP in both compilation and linkage.
target_compile_options(nbody PRIVATE -O3 -march=native -Wall -Wextra)
target_link_libraries(nbody PRIVATE OpenMP::OpenMP_CXX m)
