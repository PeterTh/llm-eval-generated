cmake_minimum_required(VERSION 3.10)
project(spmv_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)

add_executable(spmv spmv.cpp)

# Enable optimizations
if(CMAKE_CUDA_COMPILER)
    target_compile_options(spmv PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-O3 -march=native -Wall -Wextra>
                                        $<$<COMPILE_LANGUAGE:CUDA>:-O3>)
else()
    target_compile_options(spmv PRIVATE -O3 -march=native -Wall -Wextra)
endif()

set_source_files_properties(spmv.cpp PROPERTIES LANGUAGE CUDA)

# Link math library
target_link_libraries(spmv m)
