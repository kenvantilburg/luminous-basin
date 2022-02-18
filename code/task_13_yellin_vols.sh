#!/bin/bash

ip=$1

# Activate conda environment
source ~/.bashrc
conda activate luminous-basin

DIR="/mnt/home/kvantilburg/ceph/luminous-basin/results/data/vols/vols_$ip/"

if [ -d "$DIR" ]; then
  ### Take no action if $DIR exists ###
  echo "${DIR} already exists..."
else
  ###  Control will jump here if $DIR does NOT exist ###
  echo "${DIR} not found. Creating it..."
  mkdir $DIR
fi

ik=10

for im in {21..369..1}
    do
        echo "Starting volume computation $ip at mass index $im with $ik bins ..."
        python3 script_13_yellin_vols_data.py $ip $im $ik
    done

        

