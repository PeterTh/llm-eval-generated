cmake_minimum_required(VERSION 3.18)
project(roomsim_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 70 80 90)
endif()

find_package(MPI REQUIRED)

set_source_files_properties(roomsim.cpp PROPERTIES LANGUAGE CUDA)
add_executable(roomsim roomsim.cpp)

target_compile_options(roomsim PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 --extended-lambda --expt-relaxed-constexpr -Xcompiler=-fopenmp,-O3,-march=native,-Wall,-Wextra>)

target_include_directories(roomsim PRIVATE ${MPI_CXX_INCLUDE_DIRS})
target_link_libraries(roomsim ${MPI_CXX_LIBRARIES} gomp m)
