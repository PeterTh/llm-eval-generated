cmake_minimum_required(VERSION 3.10)
project(roomsim_benchmark)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)
find_package(CUDAToolkit)

if(CUDAToolkit_FOUND)
    enable_language(CUDA)
    add_definitions(-DUSE_CUDA)
    set_source_files_properties(roomsim.cpp PROPERTIES LANGUAGE CUDA)
endif()

add_executable(roomsim roomsim.cpp)

target_link_libraries(roomsim PRIVATE MPI::MPI_CXX OpenMP::OpenMP_CXX m)
target_compile_options(roomsim PRIVATE ${MPI_CXX_COMPILE_OPTIONS} ${OpenMP_CXX_FLAGS} -O3 -march=native -Wall -Wextra -Wpedantic)

if(CUDAToolkit_FOUND)
    target_link_libraries(roomsim PRIVATE CUDA::cudart)
endif()
