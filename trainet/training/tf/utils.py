"""Utils for training in Tensorflow"""

import os

import tensorflow as tf
import numpy as np


def set_global_random_seed(random_seed):
    """Set random seed for numpy and tensorflow

    :param random_seed: seed to use when setting the random seed
    :type random_seed: int
    """

    os.environ['PYTHONHASHSEED'] = random_seed
    np.random.seed(random_seed)
    tf.set_random_seed(random_seed)
