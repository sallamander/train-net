"""Dataset for applying transformations to a NumPyDataset"""

from torch.utils.data import Dataset, DataLoader

from trainet.datasets.base_dataset import NumPyDataset
from trainet.datasets.ops import apply_transformation
from trainet.utils.generic_utils import cycle


class AugmentedDataset(NumPyDataset):
    """NumPyDataset with transformations applied"""

    def __init__(self, numpy_dataset, transformations=None):
        """Init

        :param numpy_dataset: dataset that provides samples for training
        :type numpy_dataset: datasets.base_dataset.NumPyDataset
        :param transformations: holds 2 element tuples with the first element
         being a function to apply to the dataset samples and the second
         element being a dictionary of keyword arguments to pass to those
         functions
        :type transformations: list[tuple(function, dict)]
        """

        self.numpy_dataset = numpy_dataset
        self.transformations = (
            transformations if not None else []
        )

    def __getitem__(self, idx):
        """Return the transformed sample at index `idx` of `self.numpy_dataset`

        :param idx: the index of the observation to return
        :type idx: int
        :return: sample returned from `self.numpy_dataset.__getitem__`
        :rtype: dict
        """

        sample = self.numpy_dataset[idx]

        it = self.transformations
        for transformation_fn, transformation_fn_kwargs in it:
            transformation_fn_kwargs = transformation_fn_kwargs.copy()
            sample_keys = transformation_fn_kwargs.pop('sample_keys')

            sample = apply_transformation(
                transformation_fn, sample, sample_keys,
                transformation_fn_kwargs
            )

        return sample

    def __len__(self):
        """Return the size of the dataset

        :return: size of the dataset
        :rtype: int
        """

        return len(self.numpy_dataset)

    def as_generator(self, shuffle=False, n_workers=0):
        """Return a generator that yields the entire dataset once

        This method is intended to act as a lightweight wrapper around the
        torch.utils.data.DataLoader class, which has built-in shuffling of the
        data without loading it all into memory. This method purposely removes
        the added batch dimension from DataLoader such that each element
        yielded is still a single sample, just as if it came from indexing into
        this class, e.g. AugmentedDataset[10].

        :param shuffle: if True, shuffle the data before returning it
        :type shuffle: bool
        :param n_workers: number of subprocesses to use for data loading
        :type n_workers: int
        :return: generator that yields the entire dataset once
        :rtype: generator
        """

        data_loader = DataLoader(
            dataset=self, shuffle=shuffle, num_workers=n_workers
        )
        for sample in cycle(data_loader):
            sample_batch_dim_removed = {}
            for key, val in sample.items():
                sample_batch_dim_removed[key] = val[0]
            yield sample_batch_dim_removed

    def input_keys(self):
        """Return the sample keys that denote a learning algorithm's inputs

        These keys should be contained in the dictionary returned from
        __getitem__, and correspond to the keys that will be the inputs to a
        neural network.

        :return: input key names
        :rtype: list[str]
        """
        return self.numpy_dataset.input_keys

    def required_config_keys(self):
        """Return the keys required to be in the config passed to the __init__

        :return: required configuration keys
        :rtype: set{str}
        """
        return self.numpy_dataset.required_config_keys

    def sample_shapes(self):
        """Return shapes of the sample elements returned from __getitem__

        :return: dict holding tuples of the shapes for the elements of the
         sample returned from __getitem__
        :return: dict{str: tuple}
        """
        return self.numpy_dataset.sample_shapes

    def sample_types(self):
        """Return data types of the sample elements returned from __getitem__

        :return: element data types for each element in a sample returned from
         __getitem__
        :rtype: dict{str: str}
        """
        return self.numpy_dataset.sample_types

    def target_keys(self):
        """Return the sample keys that denote a learning algorithm's targets

        These keys should be contained in the dictionary returned from
        __getitem__, and correspond to the keys that will be the targets to a
        neural network.

        :return: target key names
        :rtype: list[str]
        """
        return self.numpy_dataset.target_keys
