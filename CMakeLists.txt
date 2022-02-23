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

option(BUILD_SHARED_LIBS "Build using shared libraries" ON)
message("BUILD_SHARED_LIBS: ${BUILD_SHARED_LIBS}")
# Add source to this project's executable.
#add_executable
#add_library
set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)

add_library(ComputeLib
    #"Compute/gradients.cpp"
    #"Compute/kmeans.cpp"
    #"Compute/random.cpp"
    #"Compute/lstq.cpp"
    #"Compute/linalg_utils.cpp"
    
    #"Optim/model.cpp"
    #"Optim/optim.cpp"
    #"Optim/strp.cpp"
    
    
    #"Expression/expression_nodes.cpp"
    #"Expression/expression.cpp"
    "Expression/lexer.cpp"
    "Expression/lexer_default.cpp"
    "Expression/token.cpp"
    
    
    
    #"Models/models.cpp"
)
set_target_properties(ComputeLib PROPERTIES COMPILE_PDB_NAME "ComputeLib")
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
message("CMAKE_PREFIX_PATH: ${CMAKE_PREFIX_PATH}")

# TODO: Add tests and install targets if needed.
find_package(Torch REQUIRED CONFIG)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

target_link_libraries(ComputeLib PUBLIC "${TORCH_LIBRARIES}")
target_include_directories(ComputeLib PUBLIC "${TORCH_INCLUDE_DIRS}")


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

set_target_properties(ComputeLib PROPERTIES PUBLIC_HEADER "compute.hpp")

# Tests
add_executable(ComputeTestDiffExpression "Tests/test_diff_expression.cpp")
target_link_libraries(ComputeTestDiffExpression ComputeLib)

add_executable(ComputeTestEnv "Tests/test_env.cpp")
target_link_libraries(ComputeTestEnv ComputeLib)

add_executable(ComputeTestExpMod "Tests/test_expression_model.cpp")
target_link_libraries(ComputeTestExpMod ComputeLib)

add_executable(ComputeTestLexer "Tests/test_lexer.cpp")
target_link_libraries(ComputeTestLexer ComputeLib)

add_executable(ComputeTestLinearFit "Tests/test_linear_fit.cpp")
target_link_libraries(ComputeTestLinearFit ComputeLib)

add_executable(ComputeTestLSTQ "Tests/test_lstq.cpp")
target_link_libraries(ComputeTestLSTQ ComputeLib)

add_executable(ComputeTestSTRP "Tests/test_strp.cpp")
target_link_libraries(ComputeTestSTRP ComputeLib)

# Prototyping Environments
add_executable(ComputeProtP1 "Prototyping/CPP/p1.cpp")