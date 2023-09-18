
cmake_minimum_required(VERSION 3.12)  # 设置所需的最低CMake版本
set(CMAKE_CXX_STANDARD 11)  # 可根据需要设置为其他C++标准
project(partition)  # 设置项目名称


FIND_PACKAGE( OpenMP REQUIRED)
if(OPENMP_FOUND)
message("OPENMP FOUND")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()
include_directories("./include")
include_directories("../../3rdparty/parallel-hashmap")
# target_include_directories(my_app PRIVATE include)
# 添加可执行文件
# add_executable(fennel fennel.cpp)
# add_executable(fennel_test fennel_test.cpp)
# add_executable(openmptest openmptest.cpp)
# add_executable(bloomfilter bloomfilter.cpp)
# add_executable(ogb-products-bloomfilter ogb-products-bloomfilter.cpp)
# add_executable(getEdges getEdges.cpp)
add_executable(readGraph main.cpp readGraph.cpp)
# add_executable(map_test map_test.cpp)
# add_executable(unordered_map_test unorder_map_test.cpp)
# 添加其他的源文件
# add_executable(my_app main.cpp source1.cpp source2.cpp)

# 设置C++标准


# 添加其他库和依赖
# find_package(SomePackage REQUIRED)
# target_link_libraries(my_app SomePackage::SomeLibrary)

# 设置编译选项
# target_compile_options(my_app PRIVATE -Wall -Wextra)

# 设置包含目录
# target_include_directories(my_app PRIVATE include)

# 设置链接目录
# link_directories(/path/to/library)

# 设置安装规则
# install(TARGETS my_app DESTINATION bin)



