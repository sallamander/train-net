"""Loader to create a tensorflow.data.Dataset objects for training"""

import tensorflow as tf


def format_batch(batch, input_keys, target_keys):
    """Format the batch from a single dictionary into a tuple of dictionaries

    :param batch: batch of inputs and targets
    :type batch: dict[tensorflow.Tensor]
    :param input_keys: names of the keys in `batch` that are inputs to a model
    :type input_keys: list[str]
    :param target_keys: names of the keys in `batch` that are targets for a
     model
    :type target_keys: list[str]
    :return: 2-element tuple holding the inputs and targets
    :rtype: tuple(dict)
    """

    inputs = {input_key: batch[input_key] for input_key in input_keys}
    targets = {target_key: batch[target_key] for target_key in target_keys}

    return inputs, targets


class DataLoader():
    """Loader for batches of a tf.data.Dataset"""

    def __init__(self, augmented_dataset):
        """Init

        :param augmented_dataset: dataset that provides samples for training
        :type augmented_dataset: datasets.augmented_dataset.AugmentedDataset
        """

        self.augmented_dataset = augmented_dataset

    def _calculate_sample_shapes(self):
        """Calculate the array shapes of each sample element

        Barring any one-hot encoding transformations that are specified, this
        method simply pulls the `self.augmented_dataset.sample_shapes` property
        and returns it. If there are any one-hot encodings specified, though,
        it adjusts the sample shape to (num_classes, ) as specified in the
        one-hot encoding dictionary. Target keys that are to be one-hot encoded
        will have a sample shape of (1, ) (since they are an integer), but
        after one-hot encoding tensorflow will exepect a shape of (n_classes):

        :return: shapes of each sample element that will be returned when
         training
        :rtype: dict[key: tuple]
        """

        sample_shapes = {}
        for name, shape in self.augmented_dataset.sample_shapes.items():
            for transformation in self.augmented_dataset.transformations:
                transformation_fn = transformation[0]
                if transformation_fn == tf.keras.utils.to_categorical:
                    sample_keys = transformation[2]
                    if name in sample_keys:
                        shape = (transformation[1]['num_classes'], )
            sample_shapes[name] = tf.TensorShape(shape)

        return sample_shapes

    def get_infinite_iter(self, batch_size, shuffle=False,
                          prefetch_buffer_size=1, n_workers=0):
        """Return a tf.data.Dataset that iterates over the data indefinitely

        :param batch_size: size of the batches to return
        :type batch_size: int
        :param shuffle: if True, re-shuffle the data at the end of every epoch
        :type shuffle: bool
        :param prefetch_buffer_size: number of batches to prefetch
        :type prefetch_buffer_size: int
        :param n_workers: number of subprocesses to use for data loading
        :type n_workers: int
        :return: dataset that iterates over the data indefinitely
        :rtype: tensorflow.data.Dataset
        """

        generator = self.augmented_dataset.as_generator(
            shuffle=shuffle, n_workers=n_workers
        )
        sample_shapes = self._calculate_sample_shapes()
        dataset = tf.data.Dataset.from_generator(
            lambda: generator, self.augmented_dataset.sample_types,
            sample_shapes
        )
        dataset = dataset.repeat()

        dataset = dataset.batch(batch_size)
        dataset = dataset.map(
            lambda batch: format_batch(
                batch, input_keys=self.augmented_dataset.input_keys,
                target_keys=self.augmented_dataset.target_keys
            )
        )
        dataset = dataset.prefetch(prefetch_buffer_size)
        return dataset
