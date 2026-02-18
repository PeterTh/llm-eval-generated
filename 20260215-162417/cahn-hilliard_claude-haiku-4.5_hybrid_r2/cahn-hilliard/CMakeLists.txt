cmake_minimum_required(VERSION 3.10)
project(cahn_hilliard_benchmark CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find MPI
find_package(MPI REQUIRED)

# Find OpenMP
find_package(OpenMP REQUIRED)

# Check for CUDA compiler availability
find_program(CUDA_NVCC_EXECUTABLE nvcc)
if(CUDA_NVCC_EXECUTABLE)
    message(STATUS "CUDA compiler found: ${CUDA_NVCC_EXECUTABLE}")
    find_package(CUDA)
    if(CUDA_FOUND)
        enable_language(CUDA)
        set(CMAKE_CUDA_STANDARD 17)
        add_executable(cahn_hilliard cahn_hilliard.cpp)
        set_source_files_properties(cahn_hilliard.cpp PROPERTIES LANGUAGE CUDA)
        target_compile_definitions(cahn_hilliard PRIVATE __CUDACC__)
        target_compile_options(cahn_hilliard PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3 -lineinfo>)
        target_link_libraries(cahn_hilliard PRIVATE ${CUDA_LIBRARIES})
    else()
        message(STATUS "CUDA not fully configured, building with MPI and OpenMP only")
        add_executable(cahn_hilliard cahn_hilliard.cpp)
    endif()
else()
    message(STATUS "CUDA compiler not found, building with MPI and OpenMP only")
    add_executable(cahn_hilliard cahn_hilliard.cpp)
endif()

# Enable optimizations
target_compile_options(cahn_hilliard PRIVATE -O3 -march=native -Wall -Wextra)

# Link libraries
target_link_libraries(cahn_hilliard PRIVATE MPI::MPI_CXX)
target_link_libraries(cahn_hilliard PRIVATE OpenMP::OpenMP_CXX)
target_link_libraries(cahn_hilliard PRIVATE m)
