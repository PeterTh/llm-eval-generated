cmake_minimum_required(VERSION 3.20)
project(black_scholes_benchmark LANGUAGES CXX CUDA)

find_package(MPI 3.0 REQUIRED COMPONENTS CXX)
find_package(OpenMP REQUIRED COMPONENTS CXX)

# Compile the existing translation unit as CUDA without renaming the source.
set_source_files_properties(black_scholes.cpp PROPERTIES LANGUAGE CUDA)
add_executable(black_scholes black_scholes.cpp)
set_target_properties(black_scholes PROPERTIES
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON)
target_link_libraries(black_scholes PRIVATE MPI::MPI_CXX OpenMP::OpenMP_CXX)
separate_arguments(openmp_flags NATIVE_COMMAND "${OpenMP_CXX_FLAGS}")
foreach(flag IN LISTS openmp_flags)
    target_compile_options(black_scholes PRIVATE "-Xcompiler=${flag}")
endforeach()
target_compile_options(black_scholes PRIVATE -O3 -Xcompiler=-Wall,-Wextra)
