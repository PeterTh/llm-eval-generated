cmake_minimum_required(VERSION 3.10)
project(roomsim_benchmark)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find MPI
find_package(MPI REQUIRED)

# Find OpenMP
find_package(OpenMP REQUIRED)

# Try to find CUDA but don't fail if not available
find_package(CUDA QUIET)

add_executable(roomsim roomsim.cpp)

# Enable optimizations and warnings
target_compile_options(roomsim PRIVATE -O3 -march=native -Wall -Wextra -Wpedantic)

# Link libraries
target_link_libraries(roomsim 
    PUBLIC MPI::MPI_CXX
    PUBLIC OpenMP::OpenMP_CXX
    PRIVATE m)

# Add CUDA support if available
if(CUDA_FOUND)
    target_compile_definitions(roomsim PRIVATE ENABLE_CUDA)
    target_link_libraries(roomsim PRIVATE ${CUDA_LIBRARIES})
endif()
