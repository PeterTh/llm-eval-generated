cmake_minimum_required(VERSION 3.10)
project(roomsim_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 14)

find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)

add_executable(roomsim roomsim.cpp)
target_compile_options(roomsim PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-O3 -march=native -Wall -Wextra -Wpedantic>)
target_compile_options(roomsim PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3>)
target_link_libraries(roomsim PRIVATE MPI::MPI_CXX OpenMP::OpenMP_CXX cudart m)
set_target_properties(roomsim PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
