cmake_minimum_required(VERSION 3.18)
project(cahn_hilliard_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 70 75 80 86 90)
endif()

find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)

add_executable(cahn_hilliard cahn_hilliard.cpp)

# Compile the translation unit as CUDA so it can contain both host (MPI/OpenMP) and
# device (kernel) code in the single existing source file.
set_source_files_properties(cahn_hilliard.cpp PROPERTIES LANGUAGE CUDA)

target_compile_options(cahn_hilliard PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3;--fmad=false;-Xcompiler=-fopenmp;-Xcompiler=-O3;-Xcompiler=-march=native;-Xcompiler=-Wall;-Xcompiler=-Wextra>
)

target_include_directories(cahn_hilliard PRIVATE ${MPI_CXX_INCLUDE_DIRS})
target_link_options(cahn_hilliard PRIVATE -fopenmp)
target_link_libraries(cahn_hilliard PRIVATE MPI::MPI_CXX gomp m)
