cmake_minimum_required(VERSION 3.10)
project(floydwarshall_benchmark)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)
set(CMAKE_CUDA_ARCHITECTURES 60 70 75 80 86)
enable_language(CUDA)

add_executable(floydwarshall floydwarshall.cpp)
set_target_properties(floydwarshall PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# Enable optimizations
target_compile_options(floydwarshall PRIVATE -O3 -march=native -Wall -Wextra)

# Link math, MPI, OpenMP, CUDA
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS} ${MPI_CXX_COMPILE_FLAGS} -DUSE_MPI -DUSE_OPENMP -DUSE_CUDA")
target_link_libraries(floydwarshall m ${MPI_CXX_LIBRARIES} OpenMP::OpenMP_CXX /usr/local/cuda/lib64/libcudart.so)
target_include_directories(floydwarshall PRIVATE ${MPI_CXX_INCLUDE_PATH} /usr/local/cuda/include)
