cmake_minimum_required(VERSION 3.18)
project(roomsim_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(MPI REQUIRED)

# Target CUDA architectures: Volta (V100), Ampere (A100), Hopper (H100)
set(CMAKE_CUDA_ARCHITECTURES "70;80;90")

add_executable(roomsim roomsim.cpp)
set_source_files_properties(roomsim.cpp PROPERTIES LANGUAGE CUDA)

target_compile_options(roomsim PRIVATE
    -O3 -march=native -Wall -Wextra -Wpedantic
    -Xcompiler=-fopenmp
)

target_link_libraries(roomsim PUBLIC MPI::MPI_CXX m)

# Handle OpenMP linking: use -fopenmp which works for both nvcc and g++
target_link_options(roomsim PRIVATE "-fopenmp")
