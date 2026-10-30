cmake_minimum_required(VERSION 3.10)
project(roomsim_benchmark LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)
find_package(CUDA)

add_executable(roomsim roomsim.cpp)

target_compile_options(roomsim PRIVATE -O3 -march=native -Wall -Wextra -Wpedantic)

target_include_directories(roomsim PRIVATE ${MPI_CXX_INCLUDE_DIRS} ${CUDA_INCLUDE_DIRS})
target_compile_definitions(roomsim PRIVATE USE_MPI USE_OPENMP USE_CUDA)

target_link_libraries(roomsim PRIVATE m MPI::MPI_CXX OpenMP::OpenMP_CXX)
if(CUDA_FOUND)
    target_link_libraries(roomsim PRIVATE ${CUDA_LIBRARIES} ${CUDA_CUDART_LIBRARY})
endif()
