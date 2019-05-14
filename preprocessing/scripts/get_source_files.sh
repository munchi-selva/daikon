#!/bin/bash

#
# Retrieves the scripts and data required as a starting point for working with
# daikon for MT FS 2019 exercise 5
#

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
source ${SCRIPT_DIR}/get_dirs.sh

if [ ! -d $MOSES_REP_DIR ]
then
    #
    # Clone Mathias Müller's mosesdecoder fork
    #
    git clone https://github.com/bricksdont/moses-scripts $MOSES_REP_DIR
fi

if [ ! -d $DATA_DIR ]
then
    mkdir -p $DATA_DIR
fi

#
# Download corpus files
#
pushd $DATA_DIR
corp_source=https://files.ifi.uzh.ch/cl/archiv/2019/mt19
for corpus in train dev
do
    for lang in de en
    do
        corp_file=$corpus.$lang
        if [ ! -f $corp_file ]
        then
            wget $corp_source/$corp_file
        fi
    done
done

test_file=test.de
if [ ! -f $test_file ]
then
    wget $corp_source/$test_file
fi
popd
