#!/bin/bash

#
# Identifies the directories holding scripts and data involved in preprocessing
#

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
export MOSES_REP_DIR=${SCRIPT_DIR}/moses-scripts
export MOSES_SCRIPT_DIR=${MOSES_REP_DIR}/scripts
export DATA_DIR=${SCRIPT_DIR}/../data
