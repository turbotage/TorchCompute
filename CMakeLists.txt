# CMakeList.txt : CMake project for ComputeCMake, include source and define
# project specific logic here.
#
cmake_minimum_required (VERSION 3.20)

project(Compute)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)


#set(BUILD_SHARED_LIBS ON)
#set(CMAKE_CXX_STANDARD 17)
#set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Add source to this project's executable.
#add_executable
#add_library
add_library(ComputeLib
    "Compute/gradients.cpp"
    "Compute/kmeans.cpp"
    "Compute/random.cpp"
    "Compute/lstq.cpp"
    "Compute/linalg_utils.cpp"
    
    "Optim/model.cpp"
    "Optim/optim.cpp"
    "Optim/slmp.cpp"
    
    "Expression/expression.cpp"
    "Expression/shunter.cpp" 
    "Expression/parsing.cpp"
    "Expression/lexer.cpp"
    
    "Models/models.cpp")

#set_property(TARGET Compute PROPERTY CXX_STANDARD_REQUIRED 17)

if (MSVC)
    if (CMAKE_BUILD_TYPE STREQUAL "Debug")
        set(CMAKE_PREFIX_PATH "C:/Lib/libtorch/libtorch_debug/share/cmake/Torch")
    else()
        set(CMAKE_PREFIX_PATH "C:/Lib/libtorch/libtorch_release/share/cmake/Torch")
    endif()
else()
    set(CMAKE_PREFIX_PATH "/home/turbotage/Lib/libtorch/share/cmake/Torch")
endif(MSVC)


# TODO: Add tests and install targets if needed.
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

target_link_libraries(ComputeLib PRIVATE "${TORCH_LIBRARIES}")
target_include_directories(ComputeLib PRIVATE "${TORCH_INCLUDE_DIRS}")


# The following code block is suggested to be used on Windows.
# According to https://github.com/pytorch/pytorch/issues/25457,
# the DLLs need to be copied to avoid memory errors.
if (MSVC)
  file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
  add_custom_command(TARGET ComputeLib
                     POST_BUILD
                     COMMAND ${CMAKE_COMMAND} -E copy_if_different
                     ${TORCH_DLLS}
                     $<TARGET_FILE_DIR:ComputeLib>)
endif (MSVC)


target_precompile_headers(ComputeLib PUBLIC "pch.hpp")



# Tests
add_executable(ComputeTestSLMP "Tests/test_slmp.cpp")
target_link_libraries(ComputeTestSLMP ComputeLib)