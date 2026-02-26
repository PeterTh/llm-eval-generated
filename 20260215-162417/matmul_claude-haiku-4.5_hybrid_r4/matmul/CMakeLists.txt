cmake_minimum_required(VERSION 3.10)
project(matmul_benchmark CUDA CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Set CUDA architectures before finding CUDA
if(NOT CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 60 70 80 89)
endif()

# Find MPI
find_package(MPI REQUIRED)

# Find OpenMP
find_package(OpenMP REQUIRED)

# Check for CUDA
find_package(CUDA QUIET)

# Create executable with appropriate source files
set(SOURCE_FILES matmul.cpp)

if(CUDA_FOUND)
    message(STATUS "CUDA found, enabling GPU support")
    list(APPEND SOURCE_FILES matmul_cuda.cu)
    add_executable(matmul ${SOURCE_FILES})
    target_compile_definitions(matmul PRIVATE USE_CUDA)
else()
    message(STATUS "CUDA not found, building CPU-only version with OpenMP fallback")
    add_executable(matmul ${SOURCE_FILES})
endif()

# Link MPI
target_link_libraries(matmul PUBLIC MPI::MPI_CXX)

# Link OpenMP
target_link_libraries(matmul PUBLIC OpenMP::OpenMP_CXX)

# Enable optimizations
target_compile_options(matmul PRIVATE -O3 -march=native -Wall -Wextra)

# Link math library
target_link_libraries(matmul PRIVATE m)

# Configure CUDA if available
if(CUDA_FOUND)
    target_link_libraries(matmul PRIVATE ${CUDA_LIBRARIES})
    target_include_directories(matmul PRIVATE ${CUDA_INCLUDE_DIRS})
endif()


