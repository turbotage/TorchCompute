# CMakeList.txt : CMake project for ComputeCMake, include source and define
# project specific logic here.
#
cmake_minimum_required (VERSION 3.20)


#set(CMAKE_CXX_STANDARD 17)
#set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Add source to this project's executable.
add_library (TorchCompute 
    "TorchCompute.cpp" 
    "tests.cpp"
        
        "Compute/gradients.cpp"
        "Compute/kmeans.cpp"
        "Compute/random.cpp"

        "Optim/lm.cpp" 
        "Optim/model.cpp"
        "Optim/solver.cpp"

        "Expression/expression.cpp" 
        "Expression/shunter.cpp" 
        "Expression/parsing.cpp"
    )


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

target_link_libraries(TorchCompute PUBLIC "${TORCH_LIBRARIES}")
set_property(TARGET TorchCompute PROPERTY CXX_STANDARD 17)

# The following code block is suggested to be used on Windows.
# According to https://github.com/pytorch/pytorch/issues/25457,
# the DLLs need to be copied to avoid memory errors.
if (MSVC)
  file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
  add_custom_command(TARGET TorchCompute
                     POST_BUILD
                     COMMAND ${CMAKE_COMMAND} -E copy_if_different
                     ${TORCH_DLLS}
                     $<TARGET_FILE_DIR:TorchCompute>)
endif (MSVC)


#set(OpenCV_DIR "C:/OpenCV-4.5.3/opencv/build/x64/vc15/lib")
# Find OpenCV, you may need to set OpenCV_DIR variable
# to the absolute path to the directory containing OpenCVConfig.cmake file
# via the command line or GUI
#find_package(OpenCV REQUIRED)

# Link your application with OpenCV libraries
#target_link_libraries(ComputeCMake "${OpenCV_LIBS}")


#get_target_property(__dll_dbg opencv_world IMPORTED_LOCATION_DEBUG)
#get_target_property(__dll_release opencv_world  IMPORTED_LOCATION_RELEASE)


#add_custom_command(TARGET ComputeCMake POST_BUILD        # Adds a post-build event the TARGET
#    COMMAND ${CMAKE_COMMAND} -E copy_if_different           # which executes "cmake - E copy_if_different..."
#    "$<$<CONFIG:debug>:${__dll_dbg}>$<$<CONFIG:release>:${__dll_release}>"      # <--this is in-file
#    $<TARGET_FILE_DIR:ComputeCMake>                        # <--this is out-file path
#        # another dll copy if needed here
#    COMMENT "    [1718_34_ATK] copy dlls realsense2 and opencv_world")


target_precompile_headers(TorchCompute PUBLIC "pch.hpp")