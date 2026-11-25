cmake_minimum_required(VERSION 3.20)

project(qtclustering LANGUAGES CXX CUDA)

find_package(MPI REQUIRED COMPONENTS CXX)
find_package(OpenMP REQUIRED COMPONENTS CXX)

set_source_files_properties(qtclustering.cpp PROPERTIES LANGUAGE CUDA)
add_executable(qtclustering qtclustering.cpp)
set_target_properties(qtclustering PROPERTIES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON)

separate_arguments(openmp_host_flags NATIVE_COMMAND "${OpenMP_CXX_FLAGS}")
target_compile_options(qtclustering PRIVATE -O3 --fmad=false
    -Xcompiler=-O3,-march=native,-Wall,-Wextra)
foreach(flag IN LISTS openmp_host_flags)
    target_compile_options(qtclustering PRIVATE "-Xcompiler=${flag}")
endforeach()
target_link_libraries(qtclustering PRIVATE MPI::MPI_CXX OpenMP::OpenMP_CXX)
