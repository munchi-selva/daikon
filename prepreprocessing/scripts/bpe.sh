#!/bin/bash

#
# Build Byte Pair Encoding models that can be applied to the training,
# development and test corpora
#

operations=30000
if [ -n "$1" ]
then
    operations=$1
fi

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
source ${SCRIPT_DIR}/get_dirs.sh

pushd ${DATA_DIR}
subword-nmt learn-joint-bpe-and-vocab -i train.truecased.de train.truecased.en \
    --write-vocabulary vocab.de.$operations vocab.en.$operations -s $operations -o deen.bpe.$operations

for lang in de en
do
    for corpus in train dev test
    do
        cased_file=$corpus.truecased.$lang
        bped_file=$corpus.bpe.$lang
        if [ -f $cased_file ]
        then
            subword-nmt apply-bpe -i $cased_file -o $bped_file \
                -c deen.bpe.$operations --vocabulary vocab.$lang --vocabulary-threshold 50
        fi
    done
done
popd
