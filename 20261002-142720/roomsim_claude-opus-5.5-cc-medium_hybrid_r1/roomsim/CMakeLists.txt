cmake_minimum_required(VERSION 3.18)

# CUDA 12.0 does not accept GCC > 12 as host compiler; prefer g++-12 if present
if(NOT DEFINED CMAKE_CUDA_HOST_COMPILER AND NOT DEFINED ENV{CUDAHOSTCXX})
    find_program(ROOMSIM_GXX12 g++-12)
    if(ROOMSIM_GXX12)
        set(CMAKE_CUDA_HOST_COMPILER ${ROOMSIM_GXX12} CACHE FILEPATH "CUDA host compiler")
    endif()
endif()

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES native)
endif()

project(roomsim_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(MPI REQUIRED COMPONENTS CXX)
find_package(OpenMP REQUIRED COMPONENTS CXX)
find_package(CUDAToolkit REQUIRED)

# Hybrid MPI + OpenMP + CUDA: the single source is compiled as CUDA
set_source_files_properties(roomsim.cpp PROPERTIES LANGUAGE CUDA)
add_executable(roomsim roomsim.cpp)
set_target_properties(roomsim PROPERTIES LINKER_LANGUAGE CXX)

# Enable optimizations and warnings
target_compile_options(roomsim PRIVATE
    -O3
    --expt-relaxed-constexpr
    -Xcompiler=-O3,-march=native,-fopenmp,-ffp-contract=fast,-Wall,-Wextra
)

target_include_directories(roomsim PRIVATE ${MPI_CXX_INCLUDE_DIRS})
target_compile_definitions(roomsim PRIVATE OMPI_SKIP_MPICXX MPICH_SKIP_MPICXX)

# Link MPI, OpenMP, CUDA runtime and math library
target_link_libraries(roomsim PRIVATE ${MPI_CXX_LIBRARIES} OpenMP::OpenMP_CXX CUDA::cudart m)
