cmake_minimum_required(VERSION 3.10)

project(S5P)
set(CMAKE_CXX_STANDARD 11)
include_directories(${CMAKE_CURRENT_SOURCE_DIR})
find_package(Threads REQUIRED)
FIND_PACKAGE( OpenMP REQUIRED)
if(OPENMP_FOUND)
message("OPENMP FOUND")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

set(EXECUTABLE_OUTPUT_PATH "./bin/")

include_directories("/home/bear/workspace/singleGNN/3rdparty/parallel-hashmap")
# add_subdirectory(CThreadPool)



add_executable(main main.cpp readGraph.cpp)
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
