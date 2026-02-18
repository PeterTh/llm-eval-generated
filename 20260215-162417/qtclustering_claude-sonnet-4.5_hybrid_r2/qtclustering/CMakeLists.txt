cmake_minimum_required(VERSION 3.10)

project(qtclustering LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find MPI
find_package(MPI REQUIRED)

# Find OpenMP
find_package(OpenMP REQUIRED)

# Try to find CUDA
find_package(CUDAToolkit QUIET)

if(CUDAToolkit_FOUND)
    # Check if nvcc is available
    find_program(CMAKE_CUDA_COMPILER nvcc)
    if(CMAKE_CUDA_COMPILER)
        enable_language(CUDA)
        set(CMAKE_CUDA_STANDARD 14)
        set(CMAKE_CUDA_STANDARD_REQUIRED ON)
        set_source_files_properties(qtclustering.cpp PROPERTIES LANGUAGE CUDA)
        set(CUDA_ENABLED TRUE)
        message(STATUS "CUDA enabled")
    else()
        set(CUDA_ENABLED FALSE)
        message(STATUS "CUDA Toolkit found but nvcc not available, using CPU-only version")
    endif()
else()
    set(CUDA_ENABLED FALSE)
    message(STATUS "CUDA not found, using CPU-only version")
endif()

add_executable(qtclustering qtclustering.cpp)

if(CUDA_ENABLED)
    target_compile_options(qtclustering PRIVATE 
        $<$<COMPILE_LANGUAGE:CUDA>:-O3 --use_fast_math -Xcompiler=-fopenmp>
        $<$<COMPILE_LANGUAGE:CXX>:-O3 -march=native -Wall -Wextra>
    )
    target_compile_definitions(qtclustering PRIVATE USE_CUDA)
else()
    target_compile_options(qtclustering PRIVATE -O3 -march=native -Wall -Wextra -fopenmp)
endif()

target_include_directories(qtclustering PRIVATE ${MPI_CXX_INCLUDE_DIRS})

# Link libraries
if(CUDA_ENABLED)
    target_link_libraries(qtclustering 
        MPI::MPI_CXX
        OpenMP::OpenMP_CXX
        CUDA::cudart
        m
    )
else()
    target_link_libraries(qtclustering 
        MPI::MPI_CXX
        OpenMP::OpenMP_CXX
        m
    )
endif()
