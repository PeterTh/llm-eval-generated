cmake_minimum_required(VERSION 3.10)
project(unstructured_benchmark LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)

# Try to enable CUDA
include(CheckLanguage)
check_language(CUDA)

if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    
    # We MUST tell CMake that unstructured.cpp is a CUDA file if we use CUDA syntax inside
    set_source_files_properties(unstructured.cpp PROPERTIES LANGUAGE CUDA)
    
    add_executable(unstructured unstructured.cpp)
    
    # Set standard for CUDA (C++17 is usually max for older nvcc, but let's try 17 or 20)
    set_target_properties(unstructured PROPERTIES CUDA_STANDARD 17)
    
    # Flags
    # For CUDA
    target_compile_options(unstructured PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3>)
    # For Host CXX (passed via -Xcompiler if nvcc)
    target_compile_options(unstructured PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-O3,-march=native,-Wall,-Wextra>)
    
    # OpenMP flags need to be passed correctly to host compiler
    target_compile_options(unstructured PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=${OpenMP_CXX_FLAGS}>)
    
    # Link
    target_link_libraries(unstructured PRIVATE MPI::MPI_CXX OpenMP::OpenMP_CXX)
else()
    message(WARNING "CUDA not found. Building with MPI+OpenMP only.")
    add_executable(unstructured unstructured.cpp)
    target_compile_options(unstructured PRIVATE -O3 -march=native -Wall -Wextra ${OpenMP_CXX_FLAGS})
    target_link_libraries(unstructured PRIVATE MPI::MPI_CXX OpenMP::OpenMP_CXX m)
endif()
