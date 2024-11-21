cmake_minimum_required(VERSION 3.10)
project(pose_estim)

find_package(OpenCV REQUIRED)

add_executable(pose_estim main.cpp)

target_link_libraries(pose_estim ${OpenCV_LIBS})
