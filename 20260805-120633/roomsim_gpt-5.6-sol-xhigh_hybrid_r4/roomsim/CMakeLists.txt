cmake_minimum_required(VERSION 3.18)
project(roomsim_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(MPI REQUIRED COMPONENTS CXX)
find_package(OpenMP REQUIRED COMPONENTS CXX)
find_package(CUDAToolkit REQUIRED)

add_executable(roomsim roomsim.cpp)
set_source_files_properties(roomsim.cpp PROPERTIES LANGUAGE CUDA)
set_target_properties(roomsim PROPERTIES
    CUDA_ARCHITECTURES "70;75;80;86;89;90"
    CUDA_SEPARABLE_COMPILATION OFF
)

# The benchmark always uses all three programming models. Keep IEEE division
# and square root behavior because Tau uses ceil(distance / wave_speed).
target_compile_options(roomsim PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3;--allow-unsupported-compiler;--expt-relaxed-constexpr;-Xcompiler=-O3,-march=native,-fopenmp,-Wall,-Wextra>
    $<$<COMPILE_LANGUAGE:CXX>:-O3;-march=native;-fopenmp;-Wall;-Wextra;-Wpedantic>
)

target_link_libraries(roomsim PRIVATE MPI::MPI_CXX CUDA::cudart m)

# OpenMP::OpenMP_CXX adds a bare -fopenmp compile option which nvcc does not
# accept. Its discovered runtime is linked explicitly; host compilation gets
# -fopenmp through -Xcompiler above.
target_link_libraries(roomsim PRIVATE ${OpenMP_CXX_LIBRARIES})
target_include_directories(roomsim PRIVATE ${OpenMP_CXX_INCLUDE_DIRS})
