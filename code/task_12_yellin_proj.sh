#!/bin/bash

ip=$1

# Activate conda environment
source ~/.bashrc
conda activate luminous-basin

DIR="/mnt/home/kvantilburg/ceph/luminous-basin/results/data/proj/proj_$ip/"

if [ -d "$DIR" ]; then
  ### Take no action if $DIR exists ###
  echo "${DIR} already exists..."
else
  ###  Control will jump here if $DIR does NOT exist ###
  echo "${DIR} not found. Creating it..."
  mkdir $DIR
fi

for im in {21..369..1}
    do
        echo "Starting projection $ip at mass index $im..."
        python3 script_12_yellin_proj_data.py $ip $im
    done

        

