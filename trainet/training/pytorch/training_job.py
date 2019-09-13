"""Class for running a specified pytorch training job"""

from collections import OrderedDict

import pandas as pd
import torch
from torch.utils.data import DataLoader

from trainet.datasets.augmented_dataset import AugmentedDataset
from trainet.datasets.keras_dataset import KerasImageDataset
from trainet.training.base_training_job import BaseTrainingJob
from trainet.utils.generic_utils import import_object


def format_batch(batch, input_keys, target_keys, duplicate_target_keys=None):
    """Format the batch from a list of dictionaries to a tuple of tensors

    :param batch: batch of inputs and targets
    :type batch: list[dict]
    :param input_keys: names of the keys in the elements of `batch` that are
     inputs to a model
    :type input_keys: list[str]
    :param target_keys: names of the keys in `batch` that are targets for a
     model
    :type target_keys: list[str]
    :param duplicate_target_keys: holds as keys the name of a target_key and as
     values the number of times to duplicate that target value in what is
     returned, e.g. {'label': 2} would return back 2 additional copies of the
     'label' target, resulting in 3 copies total
    :type duplicate_target_keys: dict{str: int}
    :return: 2-element tuple holding the inputs and targets
    :rtype: tuple(torch.Tensor)
    """

    assert len(input_keys) == 1, 'More than one input_key is not supported.'
    assert len(target_keys) == 1, 'More than one target_key is not supported.'

    duplicate_target_keys = (
        {} if duplicate_target_keys is None else duplicate_target_keys
    )

    target_items = []
    for target_key in target_keys:
        target_items.append((target_key, []))
        if target_key in duplicate_target_keys:
            for idx_target in range(1, duplicate_target_keys[target_key] + 1):
                target_items.append(
                    ('{}{}'.format(target_key, idx_target), [])
                )
    targets_dict = OrderedDict(target_items)

    inputs = []
    for element in batch:
        inputs.append(element[input_keys[0]])

        for target_key in target_keys:
            targets_dict[target_key].append(element[target_key])
            if target_key in duplicate_target_keys:
                n_duplicates = duplicate_target_keys[target_key]
                for idx_target in range(1, n_duplicates + 1):
                    duplicate_target_key = '{}{}'.format(
                        target_key, idx_target
                    )
                    targets_dict[duplicate_target_key].append(
                        element[target_key]
                    )

    inputs = torch.stack(inputs)
    if len(target_keys) == 1 and not duplicate_target_keys:
        targets = torch.stack(targets_dict[target_key])
    else:
        targets = []
        for target_list in targets_dict.values():
            targets.append(torch.stack(target_list))

    return inputs, targets


class TrainingJob(BaseTrainingJob):
    """Runs a training job as specified via a config"""

    def _instantiate_dataset(self, set_name):
        """Return a dataset object to be used as an iterator during training

        The dataset that is returned should be able to be directly passed into
        the `train` method of whatever trainer class is specified in
        `self.config`, as either the `train_dataset` or `validation_dataset`
        argument.

        :param set_name: set to return the dataset for, one of
         {'train', 'validation'}
        :type set_name: str
        :return: two element tuple holding an iterable over the dataset for
         `set_name`, as well as the number of batches in a single pass over the
         dataset
        :rtype: tuple
        """

        assert set_name in {'train', 'validation'}
        dataset_spec = self.config['dataset']

        fpath_df_obs_key = 'fpath_df_{}'.format(set_name)
        if fpath_df_obs_key in dataset_spec:
            fpath_df_obs = dataset_spec[fpath_df_obs_key]
            df_obs = pd.read_csv(fpath_df_obs)
            dataset_spec['init_params']['df_obs'] = df_obs

        dataset_importpath = dataset_spec['importpath']
        Dataset = import_object(dataset_importpath)

        if Dataset == KerasImageDataset:
            dataset_spec['init_params']['split'] = set_name
        dataset = Dataset(**dataset_spec['init_params'])

        transformations_key = '{}_transformations'.format(set_name)
        transformations = dataset_spec[transformations_key]
        transformations = self._parse_transformations(transformations)

        dataset = AugmentedDataset(dataset, transformations)
        loading_params = dataset_spec['{}_loading_params'.format(set_name)]
        duplicate_target_keys = loading_params.pop('duplicate_target_keys', {})

        collate_fn = (
            lambda batch: format_batch(
                batch, dataset.input_keys, dataset.target_keys,
                duplicate_target_keys
            )
        )
        dataset_gen = DataLoader(
            dataset, collate_fn=collate_fn, **loading_params
        )

        return dataset_gen
