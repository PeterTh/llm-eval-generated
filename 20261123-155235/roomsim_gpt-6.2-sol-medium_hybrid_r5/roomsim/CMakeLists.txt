cmake_minimum_required(VERSION 3.18)
project(roomsim_benchmark LANGUAGES CXX CUDA)

find_package(MPI REQUIRED COMPONENTS CXX)
find_package(OpenMP REQUIRED COMPONENTS CXX)

# Keep the existing source filename and compile it as CUDA.
set_source_files_properties(roomsim.cpp PROPERTIES LANGUAGE CUDA)
add_executable(roomsim roomsim.cpp)
set_target_properties(roomsim PROPERTIES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON)
target_compile_options(roomsim PRIVATE -x cu -O3 --fmad=false
    -Xcompiler=-O3,-march=native,-Wall,-Wextra)
# Older CMake releases only attach the OpenMP flags to CXX sources.
separate_arguments(roomsim_openmp_flags NATIVE_COMMAND "${OpenMP_CXX_FLAGS}")
foreach(flag IN LISTS roomsim_openmp_flags)
    target_compile_options(roomsim PRIVATE "-Xcompiler=${flag}")
endforeach()
target_link_libraries(roomsim PRIVATE MPI::MPI_CXX OpenMP::OpenMP_CXX m)
