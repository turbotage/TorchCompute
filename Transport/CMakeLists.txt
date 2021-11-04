# CMakeList.txt : Top-level CMake project file, do global configuration
# and include sub-projects here.
#
cmake_minimum_required (VERSION 3.20)

project(Transport)

set (CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)


add_subdirectory("gRPC/")



if (MSVC)
    if (CMAKE_BUILD_TYPE STREQUAL "Debug")
        set(CMAKE_PREFIX_PATH "C:/Lib/libtorch/libtorch_debug/share/cmake/Torch")
    else()
        set(CMAKE_PREFIX_PATH "C:/Lib/libtorch/libtorch_release/share/cmake/Torch")
    endif()
else()
    set(CMAKE_PREFIX_PATH "/home/turbotage/Lib/libtorch/share/cmake/Torch")
endif(MSVC)

add_executable(Transport
	"Transport.cpp"
    "DataManager.cpp"
    "Server.cpp"

    "Threading/ThreadPool.cpp"
)

find_package(Torch REQUIRED)
target_link_libraries(Transport PRIVATE "${TORCH_LIBRARIES}")
target_include_directories(Transport PRIVATE "${TORCH_INCLUDE_DIRS}")

target_link_libraries(Transport PRIVATE Compute)
target_link_libraries(Transport PRIVATE TransportGRPC)

target_precompile_headers(Transport PUBLIC "pch.hpp")
