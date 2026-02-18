cmake_minimum_required(VERSION 3.10)
project(roomsim_benchmark CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find MPI
find_package(MPI REQUIRED)

# Find OpenMP
find_package(OpenMP REQUIRED)

# Add executable
add_executable(roomsim roomsim.cpp)

# Try to find CUDA (optional)
find_package(CUDA QUIET)

# Link all libraries at once
if(CUDA_FOUND)
    message(STATUS "CUDA found, will use GPU acceleration if available")
    target_link_libraries(roomsim 
        PRIVATE MPI::MPI_CXX
        PRIVATE OpenMP::OpenMP_CXX
        PRIVATE ${CUDA_LIBRARIES}
        PRIVATE m
    )
else()
    message(STATUS "CUDA not found, will use MPI+OpenMP only")
    target_link_libraries(roomsim 
        PRIVATE MPI::MPI_CXX
        PRIVATE OpenMP::OpenMP_CXX
        PRIVATE m
    )
endif()

# Enable optimizations and warnings
target_compile_options(roomsim PRIVATE -O3 -march=native -Wall -Wextra -Wpedantic)

