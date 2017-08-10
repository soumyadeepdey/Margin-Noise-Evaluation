cmake_minimum_required(VERSION 2.8)
project( DRD )
find_package( OpenCV REQUIRED )
add_executable( eval evaluate.cpp folder.cpp)
target_link_libraries( eval ${OpenCV_LIBS} )

