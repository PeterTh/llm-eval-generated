cmake_minimum_required(VERSION 3.24)

project(qtclustering LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)

add_executable(qtclustering qtclustering.cpp)

# The benchmark is a single translation unit that mixes host C++ with CUDA
# kernels, so it is compiled by nvcc.
set_source_files_properties(qtclustering.cpp PROPERTIES LANGUAGE CUDA)

# Build for the GPU(s) present on the build host.
set_target_properties(qtclustering PROPERTIES CUDA_ARCHITECTURES native)

target_include_directories(qtclustering PRIVATE ${MPI_CXX_INCLUDE_DIRS})

if(NOT OpenMP_CXX_FLAGS)
    set(OpenMP_CXX_FLAGS "-fopenmp")
endif()

target_compile_options(qtclustering PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-march=native>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=${OpenMP_CXX_FLAGS}>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wextra>
)

target_link_libraries(qtclustering PRIVATE
    ${MPI_CXX_LIBRARIES}
    ${OpenMP_CXX_LIBRARIES}
    m
)
