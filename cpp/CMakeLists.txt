# Minimum CMake version required
cmake_minimum_required(VERSION 3.23)

# Project name
project(opencv_example_project)

# Find OpenCV package
find_package(OpenCV REQUIRED)

# Print some OpenCV information
message(STATUS "OpenCV library status:")
message(STATUS "    config: ${OpenCV_DIR}")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${OpenCV_LIBS}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")

# Add executable target
add_executable(example src/example.cpp)

# Link OpenCV libraries
target_link_libraries(example ${OpenCV_LIBS})
