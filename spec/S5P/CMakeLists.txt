cmake_minimum_required(VERSION 3.10)

project(MyProject)
set(CMAKE_CXX_STANDARD 11)
include_directories(${CMAKE_CURRENT_SOURCE_DIR})
find_package(Threads REQUIRED)

set(EXECUTABLE_OUTPUT_PATH "./bin/")

# add_subdirectory(CThreadPool)
# 添加可执行文件 main.cpp f1.cpp f2.cpp f3.cpp


add_executable(main main.cpp globalConfig.cpp graph.cpp StreamCluster.cpp
            ClusterPackGame.cpp ClusterGameTask.cpp Partitioner.cpp readGraph.cpp)
target_link_libraries(main PRIVATE Threads::Threads)
# target_link_libraries(main PRIVATE CThreadPool)
# add_executable(main main.cpp globalConfig.cpp graph.cpp StreamCluster.cpp
#                 ClusterGameTask.cpp ClusterPackGame.cpp Partitioner.cpp)
# 添加头文件搜索路径


# 可以将需要链接的库添加在这里
# target_link_libraries(main PRIVATE your_library)

# 可以设置编译选项，比如 C++ 标准

# set(CMAKE_CXX_STANDARD_REQUIRED ON)
# set(CMAKE_CXX_EXTENSIONS OFF)
