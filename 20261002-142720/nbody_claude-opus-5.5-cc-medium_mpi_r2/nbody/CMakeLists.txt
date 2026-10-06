cmake_minimum_required(VERSION 3.10)
project(nbody_benchmark)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(MPI REQUIRED COMPONENTS CXX)

add_executable(nbody nbody.cpp)

# Enable optimizations (-fno-math-errno lets sqrt vectorize; -ffp-contract=off gives
# strict IEEE evaluation as written, so results are independent of the rank count)
target_compile_options(nbody PRIVATE -O3 -march=native -fno-math-errno -ffp-contract=off -Wall -Wextra)

# Link MPI and math library
target_link_libraries(nbody PRIVATE MPI::MPI_CXX m)
