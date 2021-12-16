cmake_minimum_required(VERSION 3.15.3)
project(Softmax C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSION OFF)

# Include all files
include_directories("${CMAKE_SOURCE_DIR}/src")

# Targets softmax
foreach(_target
    softmax softmax_2)
    add_executable(${_target}
        "src/${_target}.cc")
    target_link_libraries(${_target})
endforeach()
