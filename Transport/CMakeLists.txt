# CMakeList.txt : Top-level CMake project file, do global configuration
# and include sub-projects here.
#
cmake_minimum_required (VERSION 3.20)

add_subdirectory("gRPC/")



add_executable(Transport
	"Transport.cpp"
)

#add_custom_target(ProtoPy "${PROTO_PY}")
