#!/usr/bin/env python
# coding=utf-8

from . import train_utils
import tensorflow as tf


res = tf.cast(tf.greater(0.7, 0.5), tf.float32)
sess=tf.Session()
