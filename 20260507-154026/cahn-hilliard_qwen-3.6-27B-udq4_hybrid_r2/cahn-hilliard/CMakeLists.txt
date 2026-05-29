cmake_minimum_required(VERSION 3.10)
project(cahn_hilliard_benchmark)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ---- hybrid MPI + OpenMP + CUDA ----
enable_language(CUDA)
find_package(MPI    REQUIRED)
find_package(OpenMP REQUIRED)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 60 70 75 80 86)
endif()

add_executable(cahn_hilliard cahn_hilliard.cpp)
set_source_files_properties(cahn_hilliard.cpp PROPERTIES LANGUAGE CUDA)

target_compile_options(cahn_hilliard PRIVATE
    -O3 -march=native -Wall -Wextra ${OpenMP_CXX_FLAGS})
target_link_libraries(cahn_hilliard MPI::MPI_CXX OpenMP::OpenMP_CXX m)

target_include_directories(cahn_hilliard PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/..)
