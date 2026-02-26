cmake_minimum_required(VERSION 3.18)
project(cahn_hilliard_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)

find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)

# Treat .cpp file as CUDA source
set_source_files_properties(cahn_hilliard.cpp PROPERTIES LANGUAGE CUDA)

add_executable(cahn_hilliard cahn_hilliard.cpp)

# Enable optimizations
target_compile_options(cahn_hilliard PRIVATE 
    $<$<COMPILE_LANGUAGE:CXX>:-O3 -march=native -Wall -Wextra>
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 -Xcompiler -O3 -Xcompiler -march=native>
)

# Link libraries
target_link_libraries(cahn_hilliard 
    PRIVATE 
    MPI::MPI_CXX 
    OpenMP::OpenMP_CXX 
    m
)

# Allow CUDA to use MPI (might need -I for mpi.h if not standard path, usually fine with target_link_libraries)
target_include_directories(cahn_hilliard PRIVATE ${MPI_CXX_INCLUDE_PATH})
