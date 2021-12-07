#!/bin/bash
# ****************************************************************************
#  run_exp.sh
# ****************************************************************************
#
#  Author:          Horst Osberger
#  Description:     Run a list of trainings and evaluations.
#                   Note that you might need to adapt params as the venv-dir !
#
#  (c) 2021 by Horst Osberger
# ****************************************************************************

set +x

# source venv
SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
echo ${SCRIPT_DIR}
source ${SCRIPT_DIR}/../venv/bin/activate

CONFIG_DIR=${SCRIPT_DIR}/exp_configs
#CONFIG_LIST="config_resnet18_transfer.json config_resnet18_scratch.json config_resnet50_transfer.json config_resnet50_scratch.json"
CONFIG_LIST="config_resnet50_transfer.json"
for i in ${CONFIG_LIST}; do 
    echo Train and validate ${CONFIG_DIR}/$i
    python ${SCRIPT_DIR}/train.py --config_file ${CONFIG_DIR}/$i
    python ${SCRIPT_DIR}/evaluate.py --config_file ${CONFIG_DIR}/$i --use_cuda
done

echo All experiments run successfully !
