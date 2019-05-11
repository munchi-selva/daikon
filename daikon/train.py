#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Authors:
# Samuel Läubli <laeubli@cl.uzh.ch>
# Mathias Müller <mmueller@cl.uzh.ch>

import os
import time
import math
import logging
import random
import threading

import numpy as np
import tensorflow as tf

from typing import List
from multiprocessing.pool import ThreadPool

from daikon import reader
from daikon import constants as C
from daikon.vocab import create_vocab, Vocabulary
from daikon.translate import translate_lines
from daikon.compgraph import define_computation_graph
from daikon.score import score

logger = logging.getLogger(__name__)


def _sample_after_epoch(reader_ids: List[reader.ReaderTuple],
                        source_vocab: Vocabulary,
                        target_vocab: Vocabulary,
                        load_from: str,
                        epoch: int) -> None:
    """
    Samples translations during training. Three sentences are picked at random,
    translated with the current model and logged.
    """
    input_lines, output_lines = zip(*random.sample(reader_ids, 3))

    input_lines = [" ".join(source_vocab.get_words(input_line)) for input_line in input_lines]
    output_lines = [" ".join(target_vocab.get_words(output_line)) for output_line in output_lines]
    translations = translate_lines(load_from=load_from, input_lines=input_lines, train_mode=True)

    logger.debug("Sampled translations after epoch %s.", epoch)
    for input_line, output_line, translation in zip(input_lines, output_lines, translations):
        logger.debug("-" * 30)
        logger.debug("Input:\t\t%s", input_line)
        logger.debug("Predicted output:\t%s", translation)
        logger.debug("Actual output:\t%s", output_line)
    logger.debug("-" * 30)


def train(source_data: str,
          target_data: str,
          epochs: int,
          batch_size: int,
          source_vocab_max_size: int,
          target_vocab_max_size: int,
          save_to: str,
          log_to: str,
          sample_after_epoch: bool,
          source_val_data: str = None,
          target_val_data: str = None,
          val_epochs: int = C.VAL_EPOCHS,
          patience: int = C.PATIENCE,
          **kwargs) -> None:
    """Trains a translation model. See argument description in `bin/daikon`."""

    # enable early stopping if validation data files were specified
    early_stopping = source_val_data is not None and target_val_data is not None

    # create folders for model and logs if they don't exist yet
    for folder in [save_to, log_to]:
        if not os.path.exists(folder):
            os.makedirs(folder)

    if early_stopping:
        val_model_dir = os.path.join(save_to, C.VALIDATION_MODEL_DIR)
        if not os.path.exists(val_model_dir):
            os.makedirs(val_model_dir)

    logger.info("Creating vocabularies.")

    # create vocabulary to map words to ids, for source and target
    source_vocab = create_vocab(source_data, source_vocab_max_size, save_to, C.SOURCE_VOCAB_FILENAME)
    target_vocab = create_vocab(target_data, target_vocab_max_size, save_to, C.TARGET_VOCAB_FILENAME)

    logger.info("Source vocabulary: %s", source_vocab)
    logger.info("Target vocabulary: %s", target_vocab)

    if early_stopping:
        # create copies of vocabulary files used for checking validation
        # data performance
        source_vocab.save(os.path.join(val_model_dir, C.SOURCE_VOCAB_FILENAME))
        target_vocab.save(os.path.join(val_model_dir, C.TARGET_VOCAB_FILENAME))

    # convert training data to list of word ids
    logger.info("Reading training data.")
    reader_ids = list(reader.read_parallel(source_data, target_data, source_vocab, target_vocab, C.MAX_LEN))

    # define computation graph
    logger.info("Building computation graph.")

    graph_components = define_computation_graph(source_vocab.size, target_vocab.size, batch_size)
    encoder_inputs, decoder_targets, decoder_inputs, loss, train_step, decoder_logits, summary = graph_components

    saver = tf.train.Saver()

    with tf.Session() as session:
        # init
        session.run(tf.global_variables_initializer())
        # write logs (@tensorboard)
        summary_writer = tf.summary.FileWriter(log_to, graph=tf.get_default_graph())

        logger.info("Starting training.")
        tic = time.time()
        num_batches = math.floor(len(reader_ids) / batch_size)

        if early_stopping:
            # initialize metrics for checking validation data performance
            best_val_loss = float("inf")
            epochs_without_improvement = 0

        # iterate over training data `epochs` times
        for epoch in range(1, epochs + 1):
            total_loss = 0.0
            total_iter = 0

            iter_tic = time.time()

            for x, y, z in reader.iterate(reader_ids, batch_size, shuffle=True):

                feed_dict = {encoder_inputs: x,
                             decoder_inputs: y,
                             decoder_targets: z}

                l, _, s = session.run([loss, train_step, summary],
                                      feed_dict=feed_dict)
                summary_writer.add_summary(s, total_iter)
                total_loss += l
                total_iter += 1
                if total_iter % C.LOGGING_INTERVAL == 0 or total_iter == num_batches:
                    iter_taken = time.time() - iter_tic
                    logger.debug("Epoch=%s, iteration=%s/%s, samples/second=%.2f", epoch, total_iter, num_batches, batch_size * C.LOGGING_INTERVAL / float(iter_taken))
                    iter_tic = time.time()
            perplexity = np.exp(total_loss / total_iter)
            logger.info("Perplexity on training data after epoch %s: %.2f", epoch, perplexity)

            save_model = True
            if early_stopping and epoch % val_epochs == 0:
                # save a copy of the current model that can be used to check
                # its performance for the validation data
                saver.save(session, os.path.join(val_model_dir, C.MODEL_FILENAME))

                # spin off a thread to call score() for the validation data
                threadPool = ThreadPool(processes = 1)
                scoreRes = threadPool.apply_async(score, (source_val_data, target_val_data, val_model_dir, True, False))
                latest_val_loss = scoreRes.get()
                logging.info("Current model perplexity on validation data: %.2f", latest_val_loss)

                if latest_val_loss < best_val_loss:
                    logging.info("Lowest perplexity on validation data achieved")
                    best_val_loss = latest_val_loss
                    epochs_without_improvement = 0
                else:
                    save_model = False
                    epochs_without_improvement += 1
                    if epochs_without_improvement >= patience:
                        logging.info("No improvement in validation data perplexity for %d epochs: terminating training", epochs_without_improvement) 
                        return

            if save_model:
                saver.save(session, os.path.join(save_to, C.MODEL_FILENAME))

            if sample_after_epoch:
                # sample from model after epoch
                thread = threading.Thread(target=_sample_after_epoch, args=[reader_ids, source_vocab, target_vocab, save_to, epoch])
                thread.start()

        taken = time.time() - tic
        m, s = divmod(taken, 60)
        h, m = divmod(m, 60)

        logger.info("Training finished. Overall time taken to train: %d:%02d:%02d" % (h, m, s))
