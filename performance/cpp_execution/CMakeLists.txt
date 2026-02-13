cmake_minimum_required(VERSION 3.20)
project(QuantAlgorithms CXX)

# C++20 with maximum optimization
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Performance flags
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native -flto -DNDEBUG")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ffast-math -funroll-loops")
endif()

# Warning flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wpedantic")

# Find Python and pybind11
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

# Include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/common)

# Build subdirectories
add_subdirectory(common)
add_subdirectory(execution_engine)
add_subdirectory(order_book)
add_subdirectory(market_making)
add_subdirectory(hedging)
add_subdirectory(bindings)

# Testing
enable_testing()
add_subdirectory(tests)

# Benchmarks
add_subdirectory(benchmarks)

# Installation
install(TARGETS cpp_bindings
        LIBRARY DESTINATION ${CMAKE_SOURCE_DIR}/../
        ARCHIVE DESTINATION ${CMAKE_SOURCE_DIR}/../)
