cmake_minimum_required(VERSION 3.10)

# Set the project name and version
project(CamCalib)

# Find OpenCV package
find_package(OpenCV REQUIRED)

# Add the executable
add_executable(cam_calib main.cpp)

# Link OpenCV to your project
target_link_libraries(cam_calib ${OpenCV_LIBS})
