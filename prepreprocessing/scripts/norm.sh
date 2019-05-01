#!/bin/bash

#
# Normalize training, development and test corpora (e.g. fix carriage returns)
#

TMP=/var/tmp
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
source ${SCRIPT_DIR}/get_dirs.sh
MOSES_NORM=${MOSES_SCRIPT_DIR}/tokenizer/normalize-punctuation.perl

pushd ${DATA_DIR}
for lang in de en
do
    test_file=test.$lang
    if [ -f $test_file ]
    then
        cat $test_file | perl -pE 's/(\^M|\r)//g' > $TMP/$test_file
        mv $TMP/$test_file .
    fi

    for corpus in train dev test
    do
        corp_file=$corpus.$lang
        normed_file=$corpus.normalized.$lang
        if [ -f $corp_file ]
        then
            cat $corp_file | sed -e "s/\r//g" | perl $MOSES_NORM > $normed_file
        fi
    done
done
popd
