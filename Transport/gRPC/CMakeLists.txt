# CMakeList.txt : Top-level CMake project file, do global configuration
# and include sub-projects here.
#
cmake_minimum_required (VERSION 3.20)


set (CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)


find_package(Protobuf REQUIRED)
find_package(gRPC CONFIG REQUIRED)

include_directories("${Protobuf_INCLUDE_DIRS}")

file(GLOB PROTO_DEF "${CMAKE_CURRENT_SOURCE_DIR}/Protos/*.proto")
message(${PROTO_DEF})

# Protobuf CPP Generation
protobuf_generate_cpp(PROTO_SRCS PROTO_HDRS ${PROTO_DEF})
# Find gRPC CPP path
find_program(GRPC_CPP_PLUGIN grpc_cpp_plugin)
# Custom build function for gRPC cpp
function(grpc_generate_cpp SRCS HDRS)
  if(NOT ARGN)
    message(SEND_ERROR "Error: PROTOBUF_GENERATE_GRPC_CPP() called without any proto files")
    return()
  endif()

  if(PROTOBUF_GENERATE_CPP_APPEND_PATH) # This variable is common for all types of output.
    # Create an include path for each file specified
    foreach(FIL ${ARGN})
      get_filename_component(ABS_FIL ${FIL} ABSOLUTE)
      get_filename_component(ABS_PATH ${ABS_FIL} PATH)
      list(FIND _protobuf_include_path ${ABS_PATH} _contains_already)
      if(${_contains_already} EQUAL -1)
          list(APPEND _protobuf_include_path -I ${ABS_PATH})
      endif()
    endforeach()
  else()
    set(_protobuf_include_path -I ${CMAKE_CURRENT_SOURCE_DIR})
  endif()

  if(DEFINED PROTOBUF_IMPORT_DIRS)
    foreach(DIR ${Protobuf_IMPORT_DIRS})
      get_filename_component(ABS_PATH ${DIR} ABSOLUTE)
      list(FIND _protobuf_include_path ${ABS_PATH} _contains_already)
      if(${_contains_already} EQUAL -1)
          list(APPEND _protobuf_include_path -I ${ABS_PATH})
      endif()
    endforeach()
  endif()

  set(${SRCS})
  set(${HDRS})
  foreach(FIL ${ARGN})
    get_filename_component(ABS_FIL ${FIL} ABSOLUTE)
    get_filename_component(FIL_WE ${FIL} NAME_WE)

    list(APPEND ${SRCS} "${CMAKE_CURRENT_BINARY_DIR}/${FIL_WE}.grpc.pb.cc")
    list(APPEND ${HDRS} "${CMAKE_CURRENT_BINARY_DIR}/${FIL_WE}.grpc.pb.h")

    add_custom_command(
      OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/${FIL_WE}.grpc.pb.cc"
             "${CMAKE_CURRENT_BINARY_DIR}/${FIL_WE}.grpc.pb.h"
      COMMAND  ${Protobuf_PROTOC_EXECUTABLE}
      ARGS --grpc_out=${CMAKE_CURRENT_BINARY_DIR}
           --plugin=protoc-gen-grpc=${GRPC_CPP_PLUGIN}
           ${_protobuf_include_path} ${ABS_FIL}
      DEPENDS ${ABS_FIL} ${Protobuf_PROTOC_EXECUTABLE}
      COMMENT "Running gRPC C++ protocol buffer compiler on ${FIL}"
      VERBATIM)
  endforeach()

  set_source_files_properties(${${SRCS}} ${${HDRS}} PROPERTIES GENERATED TRUE)
  set(${SRCS} ${${SRCS}} PARENT_SCOPE)
  set(${HDRS} ${${HDRS}} PARENT_SCOPE)
endfunction()
# Run custom gRPC cpp build function
grpc_generate_cpp(GRPC_SRCS GRPC_HDRS ${PROTO_DEF})


include_directories(${CMAKE_CURRENT_BINARY_DIR})

add_library(TransportGRPC SHARED 
  "${PROTO_SRCS}" "${PROTO_HDRS}" "${GRPC_SRCS}" "${GRPC_HDRS}"
  "TransportService.cpp"
)


target_include_directories(TransportGRPC PUBLIC "${CMAKE_CURRENT_BINARY_DIR}")


target_link_libraries(TransportGRPC PUBLIC ${Protobuf_LIBRARIES})
target_link_libraries(TransportGRPC PUBLIC gRPC::gpr gRPC::upb gRPC::grpc gRPC::grpc++)




# PYTHON

# Protobuf Python Generation
find_program(GRPC_PYTHON_PLUGIN grpc_python_plugin)

#protobuf_generate_python(PROTO_PY ${PROTO_DEF})


function(grpc_generate_python SRCS)
  if(NOT ARGN)
    message(SEND_ERROR "Error: GRPC_GENERATE_PYTHON() called without any proto files")
    return()
  endif()
  
  find_package(PythonInterp)
  if(NOT PYTHONINTERP_FOUND)
    message(SEND_ERROR "Error: Python Interpreter is not found.")
    return()
  endif()
  
  execute_process(
    COMMAND ${PYTHON_EXECUTABLE} -m grpc_tools.protoc
    ERROR_VARIABLE _pygrpc_output
  )
  
  if (NOT (${_pygrpc_output} STREQUAL "Missing input file.\n"))
    message(SEND_ERROR 
      "Error: grpcio_tools not installed\ntry: sudo pip install grpcio_tools"
    ) 
    return() 
  endif()

  if(GRPC_GENERATE_CPP_APPEND_PATH)
    # Create an include path for each file specified
    foreach(FIL ${ARGN})
      get_filename_component(ABS_FIL ${FIL} ABSOLUTE)
      get_filename_component(ABS_PATH ${ABS_FIL} PATH)
      list(FIND _protobuf_include_path ${ABS_PATH} _contains_already)
      if(${_contains_already} EQUAL -1)
          list(APPEND _protobuf_include_path -I ${ABS_PATH})
      endif()
    endforeach()
  else()
    set(_protobuf_include_path)
  endif()

  if(DEFINED PROTOBUF_IMPORT_DIRS)
    foreach(DIR ${PROTOBUF_IMPORT_DIRS})
      get_filename_component(ABS_PATH ${DIR} ABSOLUTE)
      list(FIND _protobuf_include_path ${ABS_PATH} _contains_already)
      if(${_contains_already} EQUAL -1)
          list(APPEND _protobuf_include_path -I ${ABS_PATH})
      endif()
    endforeach()
  endif()

  set(${SRCS})
  foreach(FIL ${ARGN})
    get_filename_component(ABS_FIL ${FIL} ABSOLUTE)
    get_filename_component(FIL_WE ${FIL} NAME_WE)
    get_filename_component(FIL_DIR ${ABS_FIL} DIRECTORY)

    list(APPEND _protobuf_include_path -I ${FIL_DIR})
    list(APPEND ${SRCS} "${CMAKE_CURRENT_BINARY_DIR}/${FIL_WE}_pb2.py"
      "${CMAKE_CURRENT_BINARY_DIR}/${FIL_WE}_pb2_grpc.py")

    add_custom_command(
      OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/${FIL_WE}_pb2.py" "${CMAKE_CURRENT_BINARY_DIR}/${FIL_WE}_pb2_grpc.py"
      COMMAND  ${PYTHON_EXECUTABLE} -m grpc_tools.protoc ${_protobuf_include_path}  --python_out ${CMAKE_CURRENT_BINARY_DIR} --grpc_python_out ${CMAKE_CURRENT_BINARY_DIR} ${ABS_FIL}
      DEPENDS ${ABS_FIL} ${PROTOBUF_PROTOC_EXECUTABLE}
      COMMENT "Running Python protocol buffer compiler on ${FIL} for grpc"
      VERBATIM )
  endforeach()
  set(${SRCS} ${${SRCS}} PARENT_SCOPE)
endfunction()
# Run custom gRPC python build function
grpc_generate_python(GRPC_PY ${PROTO_DEF})

#add_custom_target(PyGRPC ALL DEPENDS "${PROTO_PY}" "${GRPC_PY}")
add_custom_target(PyGRPC ALL DEPENDS "${GRPC_PY}")
