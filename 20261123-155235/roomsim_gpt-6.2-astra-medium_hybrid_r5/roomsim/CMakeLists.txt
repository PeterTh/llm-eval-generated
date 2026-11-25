cmake_minimum_required(VERSION 3.20)
project(roomsim_benchmark LANGUAGES CXX CUDA)

find_package(MPI REQUIRED COMPONENTS CXX)
find_package(OpenMP REQUIRED COMPONENTS CXX)

add_executable(roomsim roomsim.cpp)
set_source_files_properties(roomsim.cpp PROPERTIES LANGUAGE CUDA)
set_target_properties(roomsim PROPERTIES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON)
target_compile_definitions(roomsim PRIVATE OMPI_SKIP_MPICXX MPICH_SKIP_MPICXX)
target_link_libraries(roomsim PRIVATE MPI::MPI_CXX OpenMP::OpenMP_CXX m)
target_compile_options(roomsim PRIVATE -O3
    "$<$<CUDA_COMPILER_ID:NVIDIA>:-Xcompiler=${OpenMP_CXX_FLAGS},-march=native;--expt-relaxed-constexpr>"
    "$<$<CUDA_COMPILER_ID:Clang>:${OpenMP_CXX_FLAGS}>")
