cmake_minimum_required(VERSION 3.18)
project(cahn_hilliard_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Detect GPU architecture
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 70 80 89 90)
endif()

find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)

# Treat the .cpp file as CUDA source
set_source_files_properties(cahn_hilliard.cpp PROPERTIES LANGUAGE CUDA)

add_executable(cahn_hilliard cahn_hilliard.cpp)

target_include_directories(cahn_hilliard PRIVATE ${MPI_CXX_INCLUDE_DIRS})
target_link_libraries(cahn_hilliard ${MPI_CXX_LIBRARIES} OpenMP::OpenMP_CXX m)
target_compile_options(cahn_hilliard PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 --extended-lambda -Xcompiler=-fopenmp,-O3,-march=native>
)
target_compile_definitions(cahn_hilliard PRIVATE ${MPI_CXX_COMPILE_DEFINITIONS})
