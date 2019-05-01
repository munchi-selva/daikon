#!/bin/bash

#
# Extract mini training sets from the full train.XX.lang corpora
#

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
source ${SCRIPT_DIR}/get_dirs.sh

pushd ${DATA_DIR}
for lang in de en
do
    for preprocess_type in truecased bpe 
    do
        source_train_file=train.${preprocess_type}.$lang
        for dataset_size in 1000 10000 100000
        do
            mini_train_file=${source_train_file}.${dataset_size}
            head -n $dataset_size $source_train_file > $mini_train_file
        done
    done
done
popd
