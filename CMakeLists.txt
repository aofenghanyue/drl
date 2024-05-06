cmake_minimum_required(VERSION 3.12)
project(flyBallBuild)

set(CMAKE_CXX_STANDARD 14)

# 指定python版本及解释器路径
set(PYTHON_EXECUTABLE "C:\\Users\\17721\\.conda\\envs\\drl\\python.exe")
set(PYTHON_INCLUDE_DIR "C:\\Users\\17721\\.conda\\envs\\drl\\include")

# 添加源文件
file(GLOB SRC_FILES ${CMAKE_CURRENT_SOURCE_DIR}/src/cpp/*.cpp)


add_subdirectory(extern/pybind11)
pybind11_add_module(flyBall ${SRC_FILES})