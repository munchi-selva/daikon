#!/bin/bash

#
# Reverses preprocessing steps for a given file
#

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
source ${SCRIPT_DIR}/get_dirs.sh

MOSES_DECASE=${MOSES_SCRIPT_DIR}/recaser/detruecase.perl
MOSES_DETOKE=${MOSES_SCRIPT_DIR}/tokenizer/detokenizer.perl

preprocessed_file=$1

file_location=`dirname ${preprocessed_file}`
file_basename=`basename ${preprocessed_file}`
file_nobpe=${file_basename}.nobpe
file_decased=${file_basename}.ntc
file_detoked=${file_basename}.detok

pushd $file_location
sed "s/\@\@ //g" < $file_basename > $file_nobpe
cat $file_nobpe | perl $MOSES_DECASE > $file_decased
cat $file_decased | perl $MOSES_DETOKE > $file_detoked
popd
