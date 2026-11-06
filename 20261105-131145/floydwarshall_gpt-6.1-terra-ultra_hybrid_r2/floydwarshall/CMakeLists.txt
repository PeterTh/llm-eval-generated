cmake_minimum_required(VERSION 3.18)

# Keep user-supplied architectures, including cross-compilation settings.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(FLOYDWARSHALL_DEFAULT_CUDA_ARCHITECTURES ON)
endif()
project(floydwarshall_benchmark LANGUAGES CXX CUDA)

# Build native code for common accelerator generations and retain PTX for
# newer devices. CUDA's default architecture can otherwise be quite old.
if(FLOYDWARSHALL_DEFAULT_CUDA_ARCHITECTURES)
    if(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 13.0)
        set(CMAKE_CUDA_ARCHITECTURES 75 80 86 89 90 90-virtual)
    elseif(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 11.8)
        set(CMAKE_CUDA_ARCHITECTURES 60 70 75 80 86 89 90 90-virtual)
    elseif(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 11.1)
        set(CMAKE_CUDA_ARCHITECTURES 60 70 75 80 86 86-virtual)
    elseif(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 11.0)
        set(CMAKE_CUDA_ARCHITECTURES 60 70 75 80 80-virtual)
    else()
        set(CMAKE_CUDA_ARCHITECTURES 60 70 75 75-virtual)
    endif()
    set(CMAKE_CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}" CACHE STRING
        "CUDA architectures to compile for" FORCE)
endif()

find_package(MPI REQUIRED COMPONENTS CXX)
find_package(OpenMP REQUIRED COMPONENTS CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# The original source name is kept while compiling it with nvcc.
set_source_files_properties(floydwarshall.cpp PROPERTIES LANGUAGE CUDA)
add_executable(floydwarshall floydwarshall.cpp)
target_link_libraries(floydwarshall PRIVATE MPI::MPI_CXX OpenMP::OpenMP_CXX)
target_compile_options(floydwarshall PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3;-Xcompiler=${OpenMP_CXX_FLAGS}>)
