cmake_minimum_required(VERSION 3.18)
project(nbody_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(MPI REQUIRED COMPONENTS CXX)
find_package(OpenMP REQUIRED COMPONENTS CXX)

set_source_files_properties(nbody.cpp PROPERTIES LANGUAGE CUDA)
add_executable(nbody nbody.cpp)

# Keep IEEE double precision and the source's accumulation order. In particular,
# fast math and fused multiply-add change close encounters over many timesteps.
target_compile_options(nbody PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3;--fmad=false;-Xcompiler=-Wall,-Wextra>)
foreach(flag IN LISTS OpenMP_CXX_FLAGS)
    target_compile_options(nbody PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=${flag}>)
endforeach()

target_link_libraries(nbody PRIVATE MPI::MPI_CXX OpenMP::OpenMP_CXX m)
