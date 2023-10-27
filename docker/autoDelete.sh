#!/bin/bash

# 列出所有已运行的容器 ID，并逐个检查容器名称
for container_id in $(docker ps -q); do
    container_name=$(docker inspect -f '{{.Name}}' "$container_id")
    
    # 检查容器名称是否包含 "Test"（注意容器名称包含 "/" 前缀）
    if [[ "$container_name" == *"/PGcluster_"* ]]; then
        echo "Stopping and removing container with ID: $container_id"
        docker stop "$container_id"
        # docker rm "$container_id"
    fi
done

# # 列出所有已停止的容器 ID，并逐个检查容器名称
# for container_id in $(docker ps -aq); do
#     container_name=$(docker inspect -f '{{.Name}}' "$container_id")

#     # 检查容器名称是否包含 "Test"（注意容器名称包含 "/" 前缀）
#     if [[ "$container_name" == *"/PGcluster_"* ]]; then
#         echo "Removing stopped container with ID: $container_id"
#         docker rm "$container_id"
#     fi
# done