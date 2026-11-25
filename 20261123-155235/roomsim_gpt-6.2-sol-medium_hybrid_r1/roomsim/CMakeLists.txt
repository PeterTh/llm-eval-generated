cmake_minimum_required(VERSION 3.20)
project(roomsim_benchmark LANGUAGES CXX CUDA)

find_package(MPI 3.0 REQUIRED COMPONENTS CXX)
find_package(OpenMP 4.5 REQUIRED COMPONENTS CXX)

set_source_files_properties(roomsim.cpp PROPERTIES LANGUAGE CUDA)
add_executable(roomsim roomsim.cpp)
set_target_properties(roomsim PROPERTIES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON)
target_link_libraries(roomsim PRIVATE MPI::MPI_CXX OpenMP::OpenMP_CXX)

# The CUDA driver also compiles the host sections of roomsim.cpp.
separate_arguments(roomsim_openmp_flags NATIVE_COMMAND "${OpenMP_CXX_FLAGS}")
list(JOIN roomsim_openmp_flags "," roomsim_openmp_flags)
target_compile_options(roomsim PRIVATE -O3 --fmad=false
    -Xcompiler=-O3,-march=native,-Wall,-Wextra
    "-Xcompiler=${roomsim_openmp_flags}")
