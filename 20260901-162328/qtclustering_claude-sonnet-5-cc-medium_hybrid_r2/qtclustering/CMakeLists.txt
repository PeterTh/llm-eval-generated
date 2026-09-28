cmake_minimum_required(VERSION 3.18)

project(qtclustering LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT DEFINED CMAKE_CUDA_HOST_COMPILER)
    find_program(GXX12_PATH g++-12)
    if(GXX12_PATH)
        set(CMAKE_CUDA_HOST_COMPILER ${GXX12_PATH})
    endif()
endif()

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 70 75 80 86)
endif()

find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)

add_executable(qtclustering qtclustering.cpp)

# Force the single source file to be compiled as CUDA so it can contain
# both host (MPI/OpenMP) code and __global__ CUDA kernels.
set_source_files_properties(qtclustering.cpp PROPERTIES LANGUAGE CUDA)

target_compile_options(qtclustering PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 --extended-lambda -Xcompiler=-O3,-Wall,-Wextra,-fopenmp>
)

set_target_properties(qtclustering PROPERTIES
    CUDA_SEPARABLE_COMPILATION OFF
)

target_link_libraries(qtclustering PRIVATE MPI::MPI_CXX OpenMP::OpenMP_CXX m)
