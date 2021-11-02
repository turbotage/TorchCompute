# CMakeList.txt : Top-level CMake project file, do global configuration
# and include sub-projects here.
#
cmake_minimum_required (VERSION 3.20)


#if (MSVC)
#    set(CMAKE_TOOLCHAIN_FILE C:\\Users\\vikto\\source\\repos\\vcpkg\\scripts\\buildsystems\\vcpkg.cmake)
#    message("MSVC")
#else()
#	set(CMAKE_TOOLCHAIN_FILE /home/turbotage/Documents/Repos/vcpkg/scripts/buildsystems/vcpkg.cmake)
#    message("Other")
#endif()

project("TransportCompute")


include_directories("Compute")

# Include sub-projects.
add_subdirectory ("Compute")
add_subdirectory ("Transport")
