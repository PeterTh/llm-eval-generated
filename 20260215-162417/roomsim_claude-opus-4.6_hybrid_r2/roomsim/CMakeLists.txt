cmake_minimum_required(VERSION 3.18)
project(roomsim_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)

add_executable(roomsim roomsim.cpp)

# Compile .cpp as CUDA
set_source_files_properties(roomsim.cpp PROPERTIES LANGUAGE CUDA)

# CUDA architecture
set_target_properties(roomsim PROPERTIES CUDA_ARCHITECTURES "70;75;80;86;89;90")
set_target_properties(roomsim PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# Compiler flags
target_compile_options(roomsim PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 --extended-lambda -Xcompiler=-fopenmp,-march=native>
)

# Link libraries
target_link_libraries(roomsim MPI::MPI_CXX OpenMP::OpenMP_CXX m)
