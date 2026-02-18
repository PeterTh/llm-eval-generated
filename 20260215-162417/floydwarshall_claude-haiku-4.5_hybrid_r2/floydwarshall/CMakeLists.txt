cmake_minimum_required(VERSION 3.10)
project(floydwarshall_benchmark LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find MPI (always required)
find_package(MPI REQUIRED)

# Find OpenMP
find_package(OpenMP REQUIRED)

# Check for CUDA support
include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    set(HAVE_CUDA ON)
else()
    set(HAVE_CUDA OFF)
endif()

# Create executable
if(HAVE_CUDA)
    set(SOURCE_FILES floydwarshall.cpp)
    set_source_files_properties(floydwarshall.cpp PROPERTIES LANGUAGE CUDA)
    add_executable(floydwarshall ${SOURCE_FILES})
    target_compile_definitions(floydwarshall PRIVATE ENABLE_CUDA)
    set(CMAKE_CUDA_STANDARD 20)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
else()
    add_executable(floydwarshall floydwarshall.cpp)
endif()

# Link libraries
target_link_libraries(floydwarshall PUBLIC MPI::MPI_CXX OpenMP::OpenMP_CXX m)

# Enable optimizations
target_compile_options(floydwarshall PRIVATE -O3 -march=native -Wall -Wextra)

if(HAVE_CUDA)
    target_compile_options(floydwarshall PRIVATE $<$<COMPILE_LANGUAGE:CUDA>: -O3 --use_fast_math>)
endif()
