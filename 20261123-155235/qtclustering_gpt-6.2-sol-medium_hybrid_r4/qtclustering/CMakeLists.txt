cmake_minimum_required(VERSION 3.20)

project(qtclustering LANGUAGES CXX CUDA)

find_package(MPI REQUIRED COMPONENTS CXX)
find_package(OpenMP REQUIRED COMPONENTS CXX)

# Keep the existing source filename while compiling its CUDA kernels and host code.
set_source_files_properties(qtclustering.cpp PROPERTIES LANGUAGE CUDA)
add_executable(qtclustering qtclustering.cpp)
set_target_properties(qtclustering PROPERTIES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON)

target_compile_options(qtclustering PRIVATE
    -O3 --fmad=false -Xcompiler=-march=native,-Wall,-Wextra)
separate_arguments(openmp_flags NATIVE_COMMAND "${OpenMP_CXX_FLAGS}")
foreach(flag IN LISTS openmp_flags)
    target_compile_options(qtclustering PRIVATE "-Xcompiler=${flag}")
endforeach()
target_link_libraries(qtclustering PRIVATE MPI::MPI_CXX OpenMP::OpenMP_CXX m)
