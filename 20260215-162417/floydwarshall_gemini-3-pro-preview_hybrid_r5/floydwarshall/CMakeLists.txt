cmake_minimum_required(VERSION 3.10)
project(floydwarshall_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)

# Force floydwarshall.cpp to be compiled with nvcc
set_source_files_properties(floydwarshall.cpp PROPERTIES LANGUAGE CUDA)

add_executable(floydwarshall floydwarshall.cpp)

# Enable optimizations and hybrid parallel flags
target_compile_options(floydwarshall PRIVATE 
    $<$<COMPILE_LANGUAGE:CXX>:-O3 -march=native -Wall -Wextra ${OpenMP_CXX_FLAGS}>
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 -arch=native -x cu -Xcompiler=${OpenMP_CXX_FLAGS}>
)

target_include_directories(floydwarshall PRIVATE ${MPI_CXX_INCLUDE_PATH})
target_link_libraries(floydwarshall 
    PRIVATE 
    m 
    MPI::MPI_CXX 
    OpenMP::OpenMP_CXX
)

# If CUDA is found (implied by LANGUAGES CUDA), link it
set_target_properties(floydwarshall PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
