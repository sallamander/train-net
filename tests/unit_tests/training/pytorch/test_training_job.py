"""Unit tests for training.pytorch.training_job"""

import os
from unittest.mock import MagicMock
import pytest

import numpy as np
import pandas as pd
import torch

from trainet.training.pytorch.training_job import format_batch, TrainingJob


class TestFormatBatch():
    """Test format batch"""

    def test_format_batch(self):
        """Test format_batch when there is only a single input and target

        This tests format_batch in the simplest case, where there is only a
        single input key (i.e. `len(input_keys) == 1`) and a single target key
        (i.e. `len(target_keys) == 1`), and there are no
        `duplicate_target_keys`.
        """

        height, width = np.random.randint(128, 600, 2)
        batch_size = np.random.randint(2, 4)
        num_channels = 3
        image = torch.from_numpy(
            np.random.random((height, width, num_channels))
        )
        label = torch.from_numpy(np.random.randint(0, 1000, 1))

        batch = [
            {'image': image, 'label': label} for _ in range(batch_size)
        ]
        formatted_batch = format_batch(
            batch, input_keys=['image'], target_keys=['label']
        )
        assert len(formatted_batch) == 2
        assert np.array_equal(
            formatted_batch[0], torch.stack([image] * batch_size)
        )
        assert np.array_equal(
            formatted_batch[1], torch.stack([label] * batch_size)
        )

    def test_format_batch__duplicate_target_keys(self):
        """Test format_batch with duplicate target keys

        This tests the case where there is only a single input and target key,
        since that is all that is allowed at this point.
        """

        height, width = np.random.randint(128, 600, 2)
        batch_size = np.random.randint(2, 4)
        num_channels = 3
        image = torch.from_numpy(
            np.random.random((height, width, num_channels))
        )
        label = torch.from_numpy(np.random.randint(0, 1000, 1))

        batch = [
            {'image': image, 'label': label} for _ in range(batch_size)
        ]
        formatted_batch = format_batch(
            batch, input_keys=['image'], target_keys=['label'],
            duplicate_target_keys={'label': 2}
        )

        assert len(formatted_batch) == 2
        assert isinstance(formatted_batch[1], list)
        assert len(formatted_batch[1]) == 3

        assert np.array_equal(
            formatted_batch[0], torch.stack([image] * batch_size)
        )
        for target in formatted_batch[1]:
            assert np.array_equal(target, torch.stack([label] * batch_size))


class TestTrainingJob():
    """Tests for TrainingJob"""

    def _get_mock_config(self):
        """Return a mock config to use for a TrainingJob

        This is implemented as a callable method (rather than a fixture)
        because we want to alter it in several of the tests, without it
        affecting the other tests.

        :return: mock config that includes a 'dataset' key and the
         configuration necessary to test the _instantiate_dataset method
        :rtype: dict
        """

        return {'dataset': {
            'fpath_df_train': 'fpath/df/train',
            'fpath_df_validation': 'fpath/df/validation',
            'importpath': 'path/to/import',
            'init_params': {'key1': 'value1', 'key2': 'value2'},
            'train_transformations': {'key3': 'value3', 'key4': 'value4'},
            'validation_transformations': {'key5': 'value5', 'key6': 'value6'},
            'train_loading_params': {'batch_size': 2},
            'validation_loading_params': {'batch_size': 4}
        }}

    def _check_instantiate_dataset(self, training_job, set_name, monkeypatch):
        """Check the `_instantiate_dataset` method for the provided `set_name`

        This tests two things:
        - The right functions / classes are used in the method itself
        - The returned output is as expected

        `pd.read_csv` is mocked to prevent tests from relying on a saved
        DataFrame.

        :param training_job: mock training job object to test
         `_instantiate_dataset` against (the `_instantiate_dataset` is not
         mocked)
        :type training_job: unittest.mock.MagicMock
        :param set_name: set to test the method for, one of 'train',
         'validation'
        :type set_name: str
        :param monkeypatch: monkeypatch object
        :type monkeypatch: _pytest.monkeypatch.MonkeyPatch
        """

        training_job._parse_transformations = MagicMock()
        training_job._parse_transformations.return_value = (
            'return_from_parse_transformations'
        )

        mock_read_csv = MagicMock()
        mock_read_csv.return_value = 'return_from_read_csv'
        monkeypatch.setattr(pd, 'read_csv', mock_read_csv)

        MockDataSet = MagicMock()
        MockDataSet.return_value = 'return_from_dataset_init'
        mock_import_object = MagicMock()
        mock_import_object.return_value = MockDataSet
        monkeypatch.setattr(
            'trainet.training.pytorch.training_job.import_object',
            mock_import_object
        )

        mock_augmented_dataset = MagicMock()
        mock_augmented_dataset_return = MagicMock()
        mock_augmented_dataset_return.__len__ = MagicMock()
        mock_augmented_dataset_return.__len__.return_value = 10
        mock_augmented_dataset.return_value = mock_augmented_dataset_return
        monkeypatch.setattr(
            'trainet.training.pytorch.training_job.AugmentedDataset',
            mock_augmented_dataset
        )

        mock_loader = MagicMock()
        mock_loader_return = MagicMock()
        mock_loader.return_value = mock_loader_return
        monkeypatch.setattr(
            'trainet.training.pytorch.training_job.DataLoader', mock_loader
        )

        dataset_gen = training_job._instantiate_dataset(
            self=training_job, set_name=set_name
        )

        assert dataset_gen == mock_loader_return
        if set_name == 'train':
            mock_read_csv.assert_called_once_with('fpath/df/train')
            training_job._parse_transformations.assert_called_once_with(
                {'key3': 'value3', 'key4': 'value4'}
            )
            mock_augmented_dataset.assert_called_once_with(
                'return_from_dataset_init', 'return_from_parse_transformations'
            )
        else:
            mock_read_csv.assert_called_once_with('fpath/df/validation')
            training_job._parse_transformations.assert_called_once_with(
                {'key5': 'value5', 'key6': 'value6'}
            )
            mock_augmented_dataset.assert_called_once_with(
                'return_from_dataset_init', 'return_from_parse_transformations'
            )

        mock_import_object.assert_called_once_with('path/to/import')
        MockDataSet.assert_called_once_with(
            df_obs='return_from_read_csv', key1='value1', key2='value2'
        )

    def test_instantiate_dataset__errors(self):
        """Test _instantiate_dataset method when errors are expected

        Errors are expected in two cases:
        - When `set_name` is not one of 'train' or 'validation'
          (AssertionError)
        """

        training_job = MagicMock()
        training_job.config = self._get_mock_config()
        training_job._instantiate_dataset = TrainingJob._instantiate_dataset

        for bad_set_name in ['test', 'val', 't']:
            with pytest.raises(AssertionError):
                training_job._instantiate_dataset(
                    self=training_job, set_name=bad_set_name
                )

    def test_instantiate_dataset__no_errors(self, monkeypatch):
        """Test _instantiate_dataset method when no errors are expected

        This tests two things via the _check_instantiate_dataset method:
        - The right functions / classes are used in the method itself
        - The returned output is as expected

        It checks these for when `set_name` is 'train' and 'validation'.
        Additionally, this tests that when there is no `fpath_df_validation`
        specified, there is no dataset returned.

        :param monkeypatch: monkeypatch object
        :type monkeypatch: _pytest.monkeypatch.MonkeyPatch
        """

        training_job = MagicMock()
        training_job.config = self._get_mock_config()
        training_job._instantiate_dataset = (
            TrainingJob._instantiate_dataset
        )

        for set_name in ('train', 'validation'):
            self._check_instantiate_dataset(
                training_job, set_name, monkeypatch
            )
