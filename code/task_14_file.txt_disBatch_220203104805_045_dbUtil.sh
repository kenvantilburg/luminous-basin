#!/bin/bash

export DISBATCH_KVSSTCP_HOST=10.250.147.0:46607 DISBATCH_ROOT=/mnt/home/carriero/projects/disBatch/beta/disBatch

if [[ $1 == '--mon' ]]
then
    exec /cm/shared/sw/nix/state/profiles/system/nixpack-jupyter/bin/python3 ${DISBATCH_ROOT}/disbatch/dbMon.py /mnt/home/kvantilburg/luminous-basin/code/task_14_file.txt_disBatch_220203104805_045
elif [[ $1 == '--engine' ]]
then
    exec /cm/shared/sw/nix/state/profiles/system/nixpack-jupyter/bin/python3 ${DISBATCH_ROOT}/disbatch/disBatch.py "$@"
else
    exec /cm/shared/sw/nix/state/profiles/system/nixpack-jupyter/bin/python3 ${DISBATCH_ROOT}/disbatch/disBatch.py --context /mnt/home/kvantilburg/luminous-basin/code/task_14_file.txt_disBatch_220203104805_045_dbUtil.sh "$@" < /dev/null &> /mnt/home/kvantilburg/luminous-basin/code/task_14_file.txt_disBatch_220203104805_045_${BASHPID}_context_launch.log
fi
