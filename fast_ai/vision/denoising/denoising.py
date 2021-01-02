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


class Denoise(tf.keras.models.Model):

    def __init__(self, width, height, depth, filters=(128, 64)):
        super(Denoise, self).__init__()
        self.width = width
        self.height = height
        self.depth = depth
        self.filters = filters
        self.channel_dimention_index = -1

        # Build encoder
        self.encoder = [tfl.Input(shape=(self.height, self.width, self.depth))]
        for f in self.filters:
            self.encoder.append(tfl.Conv2D(f, (3, 3), activation='relu', padding='same', strides=2))
            self.encoder.append(tfl.BatchNormalization(axis=self.channel_dimention_index))
        self.encoder = tf.keras.Sequential(self.encoder)

        # Build decoder
        self.decoder = []
        for f in self.filters[::-1]:
            self.decoder.append(tfl.Conv2DTranspose(f, (3, 3), strides=2, padding='same', activation='relu'))
            self.decoder.append(tfl.BatchNormalization(axis=self.channel_dimention_index))
        # apply a single Conv2D layer used to recover the
        # original depth of the image
        self.decoder.append(tfl.Conv2DTranspose(depth, kernel_size=(3, 3), padding="same", activation="sigmoid"))
        self.decoder = tf.keras.Sequential(self.decoder)

    def call(self, inputs, training=None, mask=None):
        """
        :param inputs: A tensor or list of tensors.
        :param training: Boolean or boolean scalar tensor,
            indicating whether to run the `Network` in training mode or inference mode.
        :param mask: A mask or list of masks. A mask can be
            either a tensor or None (no mask).
        :return: A tensor if there is a single output, or
            a list of tensors if there are more than one outputs.
        """
        encoded = self.encoder(inputs)
        decoded = self.decoder(encoded)
        return decoded
