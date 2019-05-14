#!/bin/bash

#
# Runs entire chain of preprocessing steps
#

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

pushd $SCRIPT_DIR
for script in   get_source_files.sh \
                norm.sh \
                toke.sh \
                truecase.sh \
                bpe.sh
do
    source $script
done
popd $SCRIPT_DIR
