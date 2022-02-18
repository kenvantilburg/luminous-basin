#!/bin/bash

ip=$1

# Activate conda environment
source ~/.bashrc
conda activate luminous-basin

ik=10

for im in {21..369..1}
    do
        echo "Starting C_max computation $ip at mass index $im with $ik bins ..."
        python3 script_14_yellin_cmax_data.py $ip $im $ik
    done

        