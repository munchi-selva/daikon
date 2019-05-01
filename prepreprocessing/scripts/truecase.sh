#!/bin/bash

#
# Apply truecasing to training, development and test corpora
#

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
source ${SCRIPT_DIR}/get_dirs.sh
MOSES_CASE_TRAIN=${MOSES_SCRIPT_DIR}/recaser/train-truecaser.perl
MOSES_CASE=${MOSES_SCRIPT_DIR}/recaser/truecase.perl

pushd ${DATA_DIR}
for lang in de en
do
    perl $MOSES_CASE_TRAIN -corpus train.tokenized.clean.$lang -model truecase_model.$lang

    for corpus in train
    do
        perl $MOSES_CASE -model truecase_model.$lang < $corpus.tokenized.clean.$lang > $corpus.truecased.$lang
    done

    for corpus in dev test
    do
        toked_file=$corpus.tokenized.$lang
        cased_file=$corpus.truecased.$lang
        if [ -f $toked_file ]
        then
            perl $MOSES_CASE -model truecase_model.$lang < $toked_file > $cased_file
        fi
    done
done
popd
