cmake_minimum_required(VERSION 3.10)

project(qtclustering LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find OpenMP (always enabled)
find_package(OpenMP REQUIRED)

# Try to find MPI
find_package(MPI QUIET)

# Try to find CUDA separately (not as language, just as library)
include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
    set(CUDA_AVAILABLE TRUE)
    enable_language(CUDA)
else()
    set(CUDA_AVAILABLE FALSE)
endif()

# Add executable as C++ project
add_executable(qtclustering qtclustering.cpp)

# Compiler flags
target_compile_options(qtclustering PRIVATE -O3 -march=native -Wall -Wextra)

# Link OpenMP (always enabled)
target_link_libraries(qtclustering PRIVATE OpenMP::OpenMP_CXX)

# Link math library
target_link_libraries(qtclustering PRIVATE m)

# Enable MPI if found
if(MPI_FOUND)
    target_compile_definitions(qtclustering PRIVATE _MPI_ENABLED)
    target_link_libraries(qtclustering PRIVATE MPI::MPI_CXX)
    message(STATUS "MPI enabled")
else()
    message(STATUS "MPI not found - compiling without MPI support")
endif()

# Enable CUDA if available
if(CUDA_AVAILABLE)
    target_compile_definitions(qtclustering PRIVATE _CUDA_ENABLED)
    # Link CUDA runtime library
    find_package(CUDA REQUIRED)
    target_link_libraries(qtclustering PRIVATE ${CUDA_LIBRARIES})
    # Set CUDA architecture
    set_target_properties(qtclustering PROPERTIES CUDA_ARCHITECTURES "60;70;80;90")
    message(STATUS "CUDA enabled")
else()
    message(STATUS "CUDA not found - compiling without CUDA support")
endif()
