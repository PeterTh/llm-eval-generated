cmake_minimum_required(VERSION 3.10)
project(cholesky_benchmark)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(CUDA REQUIRED)

find_package(CUDA REQUIRED)

cuda_add_executable(cholesky cholesky.cpp)

# Enable optimizations
# Link CUDA and math library
target_link_libraries(cholesky ${CUDA_LIBRARIES} m)
target_include_directories(cholesky PRIVATE ${CUDA_INCLUDE_DIRS})
