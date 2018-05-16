cmake_minimum_required(VERSION 3.5)
project(ai-lab)
set(CMAKE_CXX_STANDARD 11)
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin) # exec 所在的输出目录
set(LIBRARY_OUTPUT_PATH ${PROJECT_BINARY_DIR}/lib) # lib 所在的输出目录

add_executable(bp bp.cpp)
add_executable(word2vec word2vec.cpp)
