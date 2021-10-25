# CMakeList.txt : Top-level CMake project file, do global configuration
# and include sub-projects here.
#
cmake_minimum_required (VERSION 3.20)

add_executable(Transport
	"Transport.cpp"
)

#find_package(protobuf CONFIG REQUIRED)
#target_link_libraries(Transport PRIVATE protobuf::libprotoc protobuf::libprotobuf protobuf::libprotobuf-lite)

find_package(gRPC CONFIG REQUIRED)
target_link_libraries(Transport PRIVATE gRPC::gpr gRPC::upb gRPC::grpc gRPC::grpc++)

#find_package(modules CONFIG REQUIRED)
#target_link_libraries(Transport PRIVATE re2::re2 c-ares::cares)



