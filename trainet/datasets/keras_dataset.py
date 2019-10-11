"""Dataset for working with image-based datasets from keras.datasets"""

import keras.datasets as datasets
import numpy as np
from skimage.transform import resize

from trainet.datasets.base_dataset import NumPyDataset
from trainet.utils.generic_utils import validate_config


class KerasImageDataset(NumPyDataset):
    """In-Memory Dataset

    This dataset is intended to serve as a wrapper of in-memory datasets, where
    it allows for more easily fitting these in-memory datasets into the trainet
    pipeline and enjoying all of the benefits that it offers.

    Currently it supports simple in-memory datasets intended for image
    classification problems, i.e. its limited to a single set of 3D inputs
    (n_channels, height, width) and a single set of integer targets.
    """

    def __init__(self, name, split, config):
        """Init

        `config` must contain the following keys:
        - int height: height of the images to return from the __getitem__
          method
        - int width: width of the images to return from the __getitem__ method
        - int n_classes: number of classes present in the target

        :param name: name of the dataset to load, one of
         {'mnist', 'fashion_mnist', 'cifar10', 'cifar100'}
        :type name: str
        :param split: one of 'train' or 'validation' (the test set returned
         from Keras will be used as the validation set)
        :type split: str
        :param config: specifies the configuration of the image
        :type config: dict
        """

        assert name in {'mnist', 'fashion_mnist', 'cifar10', 'cifar100'}
        assert split in {'train', 'validation'}

        validate_config(config, self.required_config_keys)
        self.config = config

        inputs, targets = self.load_dataset(name, split)
        self.inputs = inputs
        self.targets = targets

    def __getitem__(self, idx):
        """Return the input / target pair at the given idx

        :param idx: index into self.inputs / self.targets to return
        :type idx: int
        :return: dict with keys:
        - numpy.ndarray image: pixel data of the returned input
        - numpy.ndarray label: class label assigned to the returned image
        """

        image = self.inputs[idx]

        n_channels = self.inputs.shape[-1]
        target_shape = (
            self.config['height'], self.config['width'], n_channels
        )
        image = resize(image, output_shape=target_shape)
        image = image.astype(self.sample_types['image'])

        label = self.targets[idx].item()
        
        sample = {'image': image, 'label': label}
        return sample

    def __len__(self):
        """Return the size of the dataset

        :return: size of the dataset
        :rtype: int
        """

        return len(self.inputs)

    @property
    def input_keys(self):
        """Return the sample keys that denote a learning algorithm's inputs

        These keys should be contained in the dictionary returned from
        __getitem__, and correspond to the keys that will be the inputs to a
        neural network.

        :return: input key names
        :rtype: set{str}
        """
        return ['image']

    def load_dataset(self, name, split):
        """Load the specified set from the provided dataset name

        :param name: name of the dataset to load, one of
         {'mnist', 'fashion_mnist', 'cifar10', 'cifar100'}
        :type name: str
        :param split: one of 'train' or 'validation' (the test set returned
         from Keras will be used as the validation set)
        :type split: str
        :return: inputs and targets for the requested set of the requested
         dataset
        :rtype: tuple[numpy.ndarray]
        """

        dataset = getattr(datasets, name)
        (x_train, y_train), (x_test, y_test) = dataset.load_data()

        if x_train.ndim == 3:
            x_train = np.expand_dims(x_train, -1)
            x_test = np.expand_dims(x_test, -1)

        if split == 'train':
            return x_train, y_train
        else:
            return x_test, y_test

    @property
    def required_config_keys(self):
        """Return the keys required to be in the config passed to the __init__

        :return: required configuration keys
        :rtype: set{str}
        """
        return {'height', 'width', 'n_classes'}

    @property
    def sample_shapes(self):
        """Return the shapes of the outputs returned

        :return: dict holding the tuples of the shapes for the values returned
         when iterating over the dataset
        :rtype: dict{str: tuple}
        """

        n_channels = self.inputs.shape[-1]
        height = self.config['height']
        width = self.config['width']

        image_shape = (height, width, n_channels)

        n_classes = self.config['n_classes']
        label_shape = (n_classes, )

        return {'image': image_shape, 'label': label_shape}

    @property
    def sample_types(self):
        """Return data types of the sample elements returned from __getitem__

        :return: element data types for each element in a sample returned from
         __getitem__
        :rtype: dict{str: str}
        """
        return {'image': 'float32', 'label': 'uint8'}

    @property
    def target_keys(self):
        """Return the sample keys that denote a learning algorithm's targets

        These should be contained in the dictionary returned from __getitem__,
        and correspond to the keys that will be the targets to a neural
        network.

        :return: target key names
        :rtype: list[str]
        """
        return ['label']
