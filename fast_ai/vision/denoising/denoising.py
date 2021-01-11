#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging.handlers
import os

import tensorflow as tf
import tensorflow.keras.layers as tfl

PYTHON_LOGGER = logging.getLogger(__name__)
if not os.path.exists("log"):
    os.mkdir("log")
HDLR = logging.handlers.TimedRotatingFileHandler("log/denoising.log",
                                                 when="midnight", backupCount=60)
STREAM_HDLR = logging.StreamHandler()
FORMATTER = logging.Formatter("%(asctime)s %(filename)s [%(levelname)s] %(message)s")
HDLR.setFormatter(FORMATTER)
STREAM_HDLR.setFormatter(FORMATTER)
PYTHON_LOGGER.addHandler(HDLR)
PYTHON_LOGGER.addHandler(STREAM_HDLR)
PYTHON_LOGGER.setLevel(logging.DEBUG)

# Absolute path to the folder location of this python file
FOLDER_ABSOLUTE_PATH = os.path.normpath(os.path.dirname(os.path.abspath(__file__)))


class Denoise:

    @staticmethod
    def build(width, height, depth, filters=(128, 64)):
        channel_dimention_index = -1

        # Build encoder
        encoder_input = tfl.Input(shape=(height, width, depth))
        x = encoder_input
        for i, f in enumerate(filters):
            x = tfl.Conv2D(f, (3, 3), activation='relu', padding='same', strides=2, name=f"{i}_Conv2D_{f}")(x)
            x = tfl.BatchNormalization(axis=channel_dimention_index, name=f"{i}_BatchNorm_{f}")(x)
        encoder = tf.keras.Model(encoder_input, x, name="encoder")

        decoder_input = tfl.Input(shape=encoder.output_shape[1:])
        x = decoder_input
        # Build decoder
        for i, f in enumerate(filters[::-1]):
            x = tfl.Conv2DTranspose(f, (3, 3), strides=2, padding='same', activation='relu', name=f"{i}_Conv2DTranspose_{f}")(x)
            x = tfl.BatchNormalization(axis=channel_dimention_index, name=f"{i}_BatchNorm_{f}")(x)
        # apply a single Conv2D layer used to recover the
        # original depth of the image
        x = tfl.Conv2D(depth, (3, 3), padding="same", activation="sigmoid", name="Conv2DSigmoid")(x)
        decoder = tf.keras.Model(decoder_input, x, name="decoder")
        auto_encoder = tf.keras.Model(encoder_input, decoder(encoder(encoder_input)), name="autoEncoder")
        return encoder, decoder, auto_encoder
