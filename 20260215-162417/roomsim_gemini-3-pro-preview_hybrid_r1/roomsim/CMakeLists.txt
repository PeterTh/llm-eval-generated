cmake_minimum_required(VERSION 3.10)
project(roomsim_benchmark LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)

# Check for CUDA
include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    add_definitions(-DUSE_CUDA)
endif()

add_executable(roomsim roomsim.cpp)

# Enable optimizations and warnings
target_compile_options(roomsim PRIVATE 
    $<$<COMPILE_LANGUAGE:CXX>:-O3 -march=native -Wall -Wextra -Wpedantic>
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 -arch=native -Xcompiler=-Wall,-Wextra>
)

# Link math library
target_link_libraries(roomsim PRIVATE m MPI::MPI_CXX OpenMP::OpenMP_CXX)

if(CMAKE_CUDA_COMPILER)
    set_source_files_properties(roomsim.cpp PROPERTIES LANGUAGE CUDA)
endif()
