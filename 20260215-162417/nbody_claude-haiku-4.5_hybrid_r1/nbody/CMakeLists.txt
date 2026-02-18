cmake_minimum_required(VERSION 3.10)
project(nbody_benchmark LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find MPI
find_package(MPI REQUIRED)

# Find OpenMP
find_package(OpenMP REQUIRED)

add_executable(nbody nbody.cpp)

# Enable optimizations
target_compile_options(nbody PRIVATE -O3 -march=native -Wall -Wextra)

# Link libraries
target_link_libraries(nbody PRIVATE MPI::MPI_CXX)
target_link_libraries(nbody PRIVATE OpenMP::OpenMP_CXX)
target_link_libraries(nbody PRIVATE m)

# Enable MPI for compilation
target_compile_definitions(nbody PRIVATE ENABLE_MPI=1)
