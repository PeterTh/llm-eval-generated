cmake_minimum_required(VERSION 3.18)

# Build for the GPU(s) of this machine by default (sm_80 or newer required).
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.24)
        set(CMAKE_CUDA_ARCHITECTURES native)
    else()
        set(CMAKE_CUDA_ARCHITECTURES 86)
    endif()
endif()

project(qtclustering LANGUAGES CXX CUDA)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# The single source file contains the CUDA kernels
set_source_files_properties(qtclustering.cpp PROPERTIES LANGUAGE CUDA)

add_executable(qtclustering qtclustering.cpp)

# Host code keeps the original optimization flags (incl. FMA contraction,
# which the GNU dialect enables by default) so data generation is identical.
# No fast-math for device code: results must be exact.
target_compile_options(qtclustering PRIVATE
    -O3
    -Xcompiler=-O3,-march=native,-ffp-contract=fast,-Wall,-Wextra
)

# Link math library
target_link_libraries(qtclustering m)
