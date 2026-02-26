cmake_minimum_required(VERSION 3.10)
project(cholesky_benchmark CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find MPI
find_package(MPI REQUIRED)

# Find OpenMP
find_package(OpenMP REQUIRED)

# Try to find CUDA (optional for this hybrid approach)
find_package(CUDA QUIET)

add_executable(cholesky cholesky.cpp)

# Enable optimizations
target_compile_options(cholesky PRIVATE -O3 -march=native -Wall -Wextra)

# Add conditional CUDA flag if available
if(CUDA_FOUND)
    target_compile_definitions(cholesky PRIVATE CUDA_AVAILABLE)
    target_include_directories(cholesky PRIVATE ${CUDA_INCLUDE_DIRS})
    target_link_libraries(cholesky PRIVATE ${CUDA_LIBRARIES})
endif()

# Link MPI, OpenMP, and math library
target_link_libraries(cholesky PRIVATE MPI::MPI_CXX OpenMP::OpenMP_CXX m)

# Set MPI compile options
target_compile_options(cholesky PRIVATE ${MPI_CXX_COMPILE_FLAGS})
target_include_directories(cholesky PRIVATE ${MPI_CXX_INCLUDE_DIRS})

