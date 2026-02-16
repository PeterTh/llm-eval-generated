cmake_minimum_required(VERSION 3.10)
project(unstructured_benchmark)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(unstructured unstructured.cpp)

# Enable OpenMP for parallelism
find_package(OpenMP REQUIRED)
if(OpenMP_CXX_FOUND)
    # Prefer imported OpenMP target when available
    if(TARGET OpenMP::OpenMP_CXX)
        target_link_libraries(unstructured PRIVATE OpenMP::OpenMP_CXX)
    else()
        target_compile_options(unstructured PRIVATE ${OpenMP_CXX_FLAGS})
        target_link_libraries(unstructured PRIVATE ${OpenMP_CXX_LIBRARIES})
    endif()
endif()

# Enable optimizations
target_compile_options(unstructured PRIVATE -O3 -march=native -Wall -Wextra)

# Link math library
target_link_libraries(unstructured PRIVATE m)
