#!/bin/bash

#
# Tokenize training, development and test corpora
#

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
source ${SCRIPT_DIR}/get_dirs.sh
MOSES_TOKE=${MOSES_SCRIPT_DIR}/tokenizer/tokenizer.perl
MOSES_CLEAN=${MOSES_SCRIPT_DIR}/training/clean-corpus-n.perl

pushd ${DATA_DIR}
for lang in de en
do
    for corpus in train dev test
    do
        normed_file=$corpus.normalized.$lang
        toked_file=$corpus.tokenized.$lang
        if [ -f $normed_file ]
        then
            cat $normed_file | perl $MOSES_TOKE -a -q -l $lang > $toked_file
        fi
    done
done

#
# Remove lines that are too long, too short, etc. to be useful for training
#
perl $MOSES_CLEAN train.tokenized de en train.tokenized.clean 1 80
popd
