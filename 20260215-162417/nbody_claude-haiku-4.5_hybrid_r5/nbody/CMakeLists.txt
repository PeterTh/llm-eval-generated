cmake_minimum_required(VERSION 3.10)
project(nbody_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Find required packages
find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)
find_package(CUDA REQUIRED)

add_executable(nbody nbody.cpp)

# Link MPI
target_link_libraries(nbody PRIVATE MPI::MPI_CXX)

# Link OpenMP
target_link_libraries(nbody PRIVATE OpenMP::OpenMP_CXX)

# Link CUDA
target_link_libraries(nbody PRIVATE ${CUDA_LIBRARIES})
target_include_directories(nbody PRIVATE ${CUDA_INCLUDE_DIRS})

# Enable optimizations
target_compile_options(nbody PRIVATE -O3 -march=native -Wall -Wextra)

# Link math library
target_link_libraries(nbody PRIVATE m)
