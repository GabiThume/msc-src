project (lpp)

cmake_minimum_required (VERSION 2.6.2)

set (TAPKEE_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../tapkee/include/")
set (EIGEN_INCLUDE_DIR "/usr/include/eigen3/")
set (IMG_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../")
set (OPENCV_INCLUDE_DIR "/usr/include/opencv")

include_directories("${TAPKEE_INCLUDE_DIR}")
include_directories("${EIGEN_INCLUDE_DIR}")
include_directories("${IMG_INCLUDE_DIR}")
include_directories("${OPENCV_INCLUDE_DIR}")


FIND_PACKAGE(PkgConfig REQUIRED)
pkg_check_modules(OpenCV REQUIRED opencv)

add_executable (lpp lpp.cpp)
	
target_link_libraries(lpp ${CMAKE_CURRENT_SOURCE_DIR}/../classifier.cpp)

find_package(OpenCV)
include_directories(${OpenCV_INCLUDE_DIRS})
target_link_libraries(lpp ${OpenCV_LIBRARIES})


if (ARPACK_FOUND)
	target_link_libraries(lpp arpack)
endif()


