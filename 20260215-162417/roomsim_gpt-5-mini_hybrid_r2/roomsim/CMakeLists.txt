cmake_minimum_required(VERSION 3.10)
project(roomsim_benchmark LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)

# Detect CUDA availability; enable optionally
option(FORCE_ENABLE_CUDA "Force enable CUDA compilation" OFF)
if(CMAKE_CUDA_COMPILER OR FORCE_ENABLE_CUDA)
    set(USE_CUDA ON)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 14)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
else()
    set(USE_CUDA OFF)
endif()

add_executable(roomsim roomsim.cpp)

if(USE_CUDA)
    # Compile roomsim.cpp with NVCC to support kernels
    set_source_files_properties(roomsim.cpp PROPERTIES LANGUAGE CUDA)
    target_compile_definitions(roomsim PRIVATE USE_CUDA)
endif()

# Enable optimizations and warnings
target_compile_options(roomsim PRIVATE -O3 -march=native -Wall -Wextra -Wpedantic)
if(OpenMP_CXX_FOUND)
    target_link_libraries(roomsim PRIVATE OpenMP::OpenMP_CXX)
endif()

# Link MPI and math
if(USE_CUDA)
    target_link_libraries(roomsim PRIVATE MPI::MPI_CXX m cudart)
else()
    target_link_libraries(roomsim PRIVATE MPI::MPI_CXX m)
endif()

# Ensure position independent code if needed
set_target_properties(roomsim PROPERTIES POSITION_INDEPENDENT_CODE ON)
