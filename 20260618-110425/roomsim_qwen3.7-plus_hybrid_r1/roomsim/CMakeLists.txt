cmake_minimum_required(VERSION 3.18)
project(roomsim_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)

add_executable(roomsim roomsim.cpp)
set_source_files_properties(roomsim.cpp PROPERTIES LANGUAGE CUDA)

set_target_properties(roomsim PROPERTIES
    CUDA_HOST_COMPILER ${MPI_CXX_COMPILER}
    CUDA_SEPARABLE_COMPILATION OFF
)

target_compile_options(roomsim PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3;-arch=sm_86;-Xcompiler;-fopenmp>
)

target_link_libraries(roomsim MPI::MPI_CXX OpenMP::OpenMP_CXX m)
