cmake_minimum_required(VERSION 3.10)
project(nbody_benchmark)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenMP REQUIRED COMPONENTS CXX)

add_executable(nbody nbody.cpp)

# Enable optimizations; softened squared distances need no sqrt errno handling.
target_compile_options(nbody PRIVATE -O3 -march=native -fno-math-errno -Wall -Wextra)

# Link the required OpenMP runtime and math library.
target_link_libraries(nbody PRIVATE OpenMP::OpenMP_CXX m)
