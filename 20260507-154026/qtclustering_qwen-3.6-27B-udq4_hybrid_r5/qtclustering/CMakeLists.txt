cmake_minimum_required(VERSION 3.10)

project(qtclustering CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)

# Detect GPU architecture (take first GPU, fallback to 70)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    execute_process(
        COMMAND bash -c "nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d ' .'"
        OUTPUT_VARIABLE GPU_CAPABILITY
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
        RESULT_VARIABLE GPU_QUERY_RESULT
    )
    if(GPU_QUERY_RESULT EQUAL 0 AND GPU_CAPABILITY)
        set(CMAKE_CUDA_ARCHITECTURES "${GPU_CAPABILITY}" CACHE STRING "GPU arch" FORCE)
    else()
        set(CMAKE_CUDA_ARCHITECTURES 70 CACHE STRING "GPU arch" FORCE)
    endif()
endif()

# Find MPI
find_package(MPI REQUIRED)

add_executable(qtclustering qtclustering.cpp)

# Compile the .cpp file with nvcc so CUDA kernels and launch syntax are handled
set_source_files_properties(qtclustering.cpp PROPERTIES LANGUAGE CUDA)

# Host compiler settings (nvcc delegates C++ compilation to this)
set_target_properties(qtclustering PROPERTIES
    CUDA_HOST_COMPILER ${CMAKE_CXX_COMPILER}
)

target_compile_options(qtclustering PRIVATE
    -O3 -march=native -Wall -Wextra -Xcompiler=-fopenmp
)

target_include_directories(qtclustering PRIVATE ${MPI_CXX_INCLUDE_PATH})

target_link_libraries(qtclustering
    MPI::MPI_CXX
    ${MPI_CXX_LIBRARIES}
    m
    -fopenmp
)
