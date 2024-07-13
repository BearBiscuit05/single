#!/bin/bash

# List all the container ids that have been run and check the container names one by one
for container_id in $(docker ps -q); do
    container_name=$(docker inspect -f '{{.Name}}' "$container_id")
    
    # Check that the container name contains "Test" (note that the container name contains a "/" prefix)
    if [[ "$container_name" == *"/PGcluster_"* ]]; then
        echo "Stopping and removing container with ID: $container_id"
        docker stop "$container_id"
        # docker rm "$container_id"
    fi
done

# # List all the container ids that have been run and check the container names one by one
# for container_id in $(docker ps -aq); do
#     container_name=$(docker inspect -f '{{.Name}}' "$container_id")

#     # Check that the container name contains "Test" (note that the container name contains a "/" prefix)
#     if [[ "$container_name" == *"/PGcluster_"* ]]; then
#         echo "Removing stopped container with ID: $container_id"
#         docker rm "$container_id"
#     fi
# done