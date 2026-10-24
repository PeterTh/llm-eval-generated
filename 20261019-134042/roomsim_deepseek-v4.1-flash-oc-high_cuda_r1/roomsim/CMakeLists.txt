cmake_minimum_required(VERSION 3.24)
project(roomsim_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Build for the architecture of the GPU present on the machine. Use the native
# architecture so the generated code matches the available hardware.
set(CMAKE_CUDA_ARCHITECTURES native)

add_executable(roomsim roomsim.cpp)

# The benchmark keeps a single source file (no new files are introduced), so
# compile it as CUDA to enable the GPU implementation.
set_source_files_properties(roomsim.cpp PROPERTIES LANGUAGE CUDA)

find_package(Threads REQUIRED)

# Enable optimizations and warnings. Host-side code (the RNG producer) must be
# optimized too, so the flags are explicitly forwarded to the host compiler.
target_compile_options(roomsim PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3;-Xcompiler=-O3;-Xcompiler=-march=native>
    $<$<COMPILE_LANGUAGE:CXX>:-O3;-march=native;-Wall;-Wextra;-Wpedantic>
)

# Link math library and the C++ threading runtime
target_link_libraries(roomsim m Threads::Threads)
