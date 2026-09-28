cmake_minimum_required(VERSION 3.18)

# CUDA 12.x officially supports host GCC <= 12; prefer g++-12 when present.
if(NOT DEFINED CMAKE_CUDA_HOST_COMPILER AND NOT DEFINED ENV{CUDAHOSTCXX})
    find_program(QT_CUDA_HOST_GXX g++-12)
    if(QT_CUDA_HOST_GXX)
        set(CMAKE_CUDA_HOST_COMPILER ${QT_CUDA_HOST_GXX})
    endif()
endif()

project(qtclustering LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES OR CMAKE_CUDA_ARCHITECTURES STREQUAL "")
    set(CMAKE_CUDA_ARCHITECTURES native)
endif()

find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)

# The single source file contains both host (MPI/OpenMP) and CUDA kernel
# code; compile it with the CUDA toolchain.
set_source_files_properties(qtclustering.cpp PROPERTIES LANGUAGE CUDA)

add_executable(qtclustering qtclustering.cpp)

# -fmad=false keeps device floating point at plainly rounded IEEE operations
# for reproducible, well-defined tie-breaking in the distance comparisons.
target_compile_options(qtclustering PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 -fmad=false -Xcompiler=-O3,-march=native,-fopenmp,-Wall,-Wextra>
    $<$<COMPILE_LANGUAGE:CXX>:-O3 -march=native -fopenmp -Wall -Wextra>)

target_include_directories(qtclustering PRIVATE ${MPI_CXX_INCLUDE_DIRS})

# Link math library, MPI, and OpenMP runtime (gomp linked directly since the
# final link may be driven by nvcc, which does not accept -fopenmp).
target_link_libraries(qtclustering ${MPI_CXX_LIBRARIES} ${OpenMP_CXX_LIBRARIES} m)
