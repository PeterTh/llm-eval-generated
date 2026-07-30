cmake_minimum_required(VERSION 3.10)
project(floydwarshall_benchmark)

enable_language(CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Detect GPU architecture for compilation
set(GPU_ARCH "80")  # Default fallback (Turing/Ampere)
execute_process(
    COMMAND bash -c "nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d ' '"
    OUTPUT_VARIABLE GPU_COMPUTE_CAP
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE NVRESULT
    ERROR_QUIET
)

if(NVRESULT EQUAL 0 AND GPU_COMPUTE_CAP)
    string(REPLACE "." ";" CC_PARTS "${GPU_COMPUTE_CAP}")
    list(GET CC_PARTS 0 CC_MAJOR)
    set(GPU_ARCH "${CC_MAJOR}0")
    message(STATUS "Detected GPU compute capability ${GPU_COMPUTE_CAP}, compiling for sm_${GPU_ARCH}")
else()
    message(STATUS "Could not detect GPU, defaulting to sm_${GPU_ARCH}")
endif()

set(CMAKE_CUDA_ARCHITECTURES ${GPU_ARCH})

add_executable(floydwarshall floydwarshall.cpp)
set_source_files_properties(floydwarshall.cpp PROPERTIES LANGUAGE CUDA)

# Enable optimizations
target_compile_options(floydwarshall PRIVATE -O3 -march=native -Wall -Wextra)
target_compile_options(floydwarshall PRIVATE -Xcompiler=-O3)

# Link math library
target_link_libraries(floydwarshall m)
