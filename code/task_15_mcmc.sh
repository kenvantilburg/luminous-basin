#!/bin/bash

im=$1

# Activate conda environment
source ~/.bashrc
conda activate luminous-basin

echo "Starting MCMC at mass index $im ..."
python3 script_15_mcmc_data.py $im
echo "Finished MCMC at mass index $im."
        

