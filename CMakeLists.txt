# CMakeList.txt : Top-level CMake project file, do global configuration
# and include sub-projects here.
#
cmake_minimum_required (VERSION 3.20)

project ("TorchCompute")

# Include sub-projects.
add_subdirectory ("Transport")
add_subdirectory ("TorchCompute")
