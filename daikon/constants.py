#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Authors:
# Samuel Läubli <laeubli@cl.uzh.ch>
# Mathias Müller <mmueller@cl.uzh.ch>

# special symbols as IDs
PAD_ID = 0
EOS_ID = 1
BOS_ID = EOS_ID
UNK_ID = 2

# special symbols as strings
PAD = '<pad>'
EOS = '<eos>'
BOS = EOS
UNK = '<unk>'

# default names of files and directories
MODEL_FILENAME = 'model'
SOURCE_VOCAB_FILENAME = 'vocab.source.json'
TARGET_VOCAB_FILENAME = 'vocab.target.json'
TRAINING_LOG_FILENAME = 'training.log'

# directory housing model for checking performance on validation data
VALIDATION_MODEL_DIR = 'val'

# max number of tokens per sequence, sentences that are
# longer than that are discarded for training
MAX_LEN = 50
# only the first SCORE_MAX_LEN tokens in a sequence are
# scored (only applies to scoring)
SCORE_MAX_LEN = 1000
# do not produce a translation that is longer than
# TRANSLATION_MAX_LEN (only applies to translation)
TRANSLATION_MAX_LEN = MAX_LEN * 2

# maximum number of different words, every additional
# word is treated as UNK
SOURCE_VOCAB_SIZE = 50000
TARGET_VOCAB_SIZE = 50000

EMBEDDING_SIZE = 512
# size of LSTM hidden state vectors
HIDDEN_SIZE = 1024

LEARNING_RATE = 0.0001

# epochs between validation data performance checks
VAL_EPOCHS = 1

# numer of models with no validation data performance improvement before
# training terminates
PATIENCE = 5

# log training progress every X batches
LOGGING_INTERVAL = 1000
