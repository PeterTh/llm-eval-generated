cmake_minimum_required(VERSION 3.18)
project(roomsim_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(MPI REQUIRED COMPONENTS CXX)
find_package(OpenMP REQUIRED COMPONENTS CXX)

# The existing source contains both the host code and CUDA kernels.
set_source_files_properties(roomsim.cpp PROPERTIES LANGUAGE CUDA)
add_executable(roomsim roomsim.cpp)
target_link_libraries(roomsim PRIVATE MPI::MPI_CXX OpenMP::OpenMP_CXX)
# OpenMP's imported target supplies its host link flags, while nvcc needs the
# host compile flag explicitly forwarded for a CUDA translation unit.
target_compile_options(roomsim PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3;--fmad=false>)
separate_arguments(_openmp_host_flags NATIVE_COMMAND "${OpenMP_CXX_FLAGS}")
foreach(_flag IN LISTS _openmp_host_flags)
    target_compile_options(roomsim PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=${_flag}>")
endforeach()
