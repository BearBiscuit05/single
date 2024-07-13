#!/bin/bash
set -e
# Parameter setting
cluster_size=32
base_hostname="PGcluster_"
network_name="pg-network"
start_ip="43.0.0."  # start IP addr
image_name="pgimage:v3"
beg_ip=8

# Create a Docker network

for i in $(seq 1 "$cluster_size"); do
    hostname="$base_hostname$i"
    ip_octet=$(($beg_ip+i))
    ip="$start_ip$ip_octet"
    custom_hosts_params+=" --add-host $hostname:$ip"
done

# Iterate to create the cluster container

for i in $(seq 1 "$cluster_size"); do
    hostname="$base_hostname$i"
    ip_octet=$(($beg_ip+i))
    ip="$start_ip$ip_octet"

    if docker ps -a | grep -q "$hostname"; then
        echo "Container '$hostname' exists."
        docker stop $hostname
        docker rm $hostname
    fi

   # Create a container and specify a host name and a custom host mapping
    docker run -d -it --name "$hostname" --ulimit nofile=65535 --hostname "$hostname" -v /raid/bear/dockerfile:/data -v /home/bear/workspace/PowerGraph:/PowerGraph --net $network_name --ip $ip $image_name /bin/bash -c "service ssh restart && /bin/bash"
    # echo "create cmd: docker run -dt --name "$hostname" --hostname "$hostname" -v $data_path:/home  $custom_hosts_params  --net $network_name --ip $ip $image_name /bin/bash"
    echo "Container pg_$i created with hostname $hostname with IP:$ip"
done

