cmake_minimum_required(VERSION 3.18)
project(roomsim_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)

# Compile .cpp as CUDA so nvcc handles __global__ kernels
set_source_files_properties(roomsim.cpp PROPERTIES LANGUAGE CUDA)

add_executable(roomsim roomsim.cpp)

target_include_directories(roomsim PRIVATE ${MPI_CXX_INCLUDE_DIRS})

# CUDA compile options: pass host flags through -Xcompiler
target_compile_options(roomsim PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        -O3
        --expt-relaxed-constexpr
        --expt-extended-lambda
        -Xcompiler=-fopenmp,-O3,-march=native,-Wall
    >
)

# Link MPI, OpenMP, and math libraries
target_link_libraries(roomsim ${MPI_CXX_LIBRARIES} -lgomp m)
