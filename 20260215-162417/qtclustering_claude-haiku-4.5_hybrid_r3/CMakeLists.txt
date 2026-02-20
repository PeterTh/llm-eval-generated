cmake_minimum_required(VERSION 3.10)

project(qtclustering CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find required packages
find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)

# Try to find CUDA (optional, will fall back to CPU implementations)
find_package(CUDA)

# Create executable
add_executable(qtclustering qtclustering/qtclustering.cpp)

# Add MPI support
target_include_directories(qtclustering PRIVATE ${MPI_CXX_INCLUDE_DIRS})
target_link_libraries(qtclustering ${MPI_CXX_LIBRARIES})
target_compile_options(qtclustering PRIVATE ${MPI_CXX_COMPILE_FLAGS})

# Add OpenMP support
target_link_libraries(qtclustering OpenMP::OpenMP_CXX)

# Add CUDA support if available
if(CUDA_FOUND)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    target_link_libraries(qtclustering ${CUDA_LIBRARIES})
    target_include_directories(qtclustering PRIVATE ${CUDA_INCLUDE_DIRS})
    target_compile_definitions(qtclustering PRIVATE CUDA_ENABLED=1)
else()
    target_compile_definitions(qtclustering PRIVATE CUDA_ENABLED=0)
endif()

# Add compilation flags
target_compile_options(qtclustering PRIVATE -O3 -march=native -Wall -Wextra)

# Link math library
target_link_libraries(qtclustering m)

# Add include directories
target_include_directories(qtclustering PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
