cmake_minimum_required(VERSION 3.18)
project(cahn_hilliard_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(MPI REQUIRED COMPONENTS CXX)
find_package(OpenMP REQUIRED COMPONENTS CXX)
find_path(MPI_HEADER_DIR mpi.h HINTS ${MPI_CXX_INCLUDE_DIRS}
  PATHS /usr/lib/${CMAKE_LIBRARY_ARCHITECTURE}/openmpi/include
        /usr/lib/${CMAKE_LIBRARY_ARCHITECTURE}/mpich/include
  PATH_SUFFIXES openmpi/include mpich/include REQUIRED)
find_library(MPI_C_LIBRARY NAMES mpi HINTS /usr/lib/${CMAKE_LIBRARY_ARCHITECTURE}/openmpi/lib
  /usr/lib/${CMAKE_LIBRARY_ARCHITECTURE}/mpich/lib REQUIRED)

add_executable(cahn_hilliard cahn_hilliard.cpp)
set_source_files_properties(cahn_hilliard.cpp PROPERTIES LANGUAGE CUDA)
set_target_properties(cahn_hilliard PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.24)
  set_property(TARGET cahn_hilliard PROPERTY CUDA_ARCHITECTURES native)
endif()
target_compile_options(cahn_hilliard PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3> $<$<COMPILE_LANGUAGE:CXX>:-O3;-march=native;-Wall;-Wextra>)
target_include_directories(cahn_hilliard PRIVATE ${MPI_HEADER_DIR})
target_compile_options(cahn_hilliard PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-fopenmp>)
target_link_libraries(cahn_hilliard PRIVATE ${MPI_C_LIBRARY} OpenMP::OpenMP_CXX m)
