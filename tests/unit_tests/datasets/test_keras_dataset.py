"""Unit tests for dataset.keras_dataset"""

from unittest.mock import MagicMock

import numpy as np
import pytest

from trainet.datasets.keras_dataset import KerasImageDataset


class TestKerasImageDataset():
    """Test KerasImageDataset"""

    @pytest.fixture(scope='class')
    def dataset_config(self):
        """dataset_config object fixture

        :return: dataset_config to be used to instantiate a ToyImageDataset
        :rtype: dict
        """

        return {'height': 64, 'width': 64}

    def test_init(self, dataset_config, monkeypatch):
        """Test __init__

        :param dataset_config: dataset_config object fixture
        :type dataset_config: dict
        :param monkeypatch: monkeypatch object
        :type monkeypatch: _pytest.monkeypatch.MonkeyPatch
        """

        mock_validate_config = MagicMock()
        monkeypatch.setattr(
            'trainet.datasets.keras_dataset.validate_config',
            mock_validate_config
        )

        mock_load_dataset = MagicMock()
        ones = np.ones((8, 64, 64, 1))
        zeros = np.zeros((10,))
        mock_load_dataset.return_value = (ones, zeros)
        monkeypatch.setattr(
            'trainet.datasets.keras_dataset.KerasImageDataset.load_dataset',
            mock_load_dataset
        )

        names_list = ['mnist', 'fashion_mnist', 'cifar10', 'cifar100']
        n_classes_list = [10, 10, 10, 1000]

        for name, n_classes in zip(names_list, n_classes_list):
            config = dataset_config.copy()
            config['n_classes'] = n_classes
            for split in ['train', 'validation']:
                dataset = KerasImageDataset(name, split, config)

                assert dataset.config == config
                assert np.array_equal(dataset.inputs, ones)
                assert np.array_equal(dataset.targets, zeros)
                mock_load_dataset.assert_called_with(name, split)
                mock_load_dataset.reset_mock()

        with pytest.raises(AssertionError):
                dataset = KerasImageDataset('bad_name', 'train', config)

        with pytest.raises(AssertionError):
                dataset = KerasImageDataset('mnist', 'bad_split', config)

    def test_getitem(self, dataset_config):
        """Test __getitem__ method

        :param dataset_config: dataset_config object fixture
        :type dataset_config: dict
        """

        mock_dataset = MagicMock()
        mock_dataset.config = dataset_config
        mock_dataset.sample_types = {'image': 'float32', 'label': 'unit8'}

        mock_inputs = np.ones((8, 32, 32, 1))
        mock_targets = np.array([
            np.array(idx) for idx in np.arange(8)
        ])
        mock_dataset.inputs = mock_inputs
        mock_dataset.targets = mock_targets

        mock_dataset.__getitem__ = KerasImageDataset.__getitem__

        sample = mock_dataset[1]
        image, label = sample['image'], sample['label']

        assert image.shape == (64, 64, 1)
        assert np.all(image == 1)
        assert label == 1

    def test_load_dataset(self, monkeypatch):
        """Test load_dataset

        :param monkeypatch: monkeypatch object
        :type monkeypatch: _pytest.monkeypatch.MonkeyPatch
        """

        keras_dataset = MagicMock()
        keras_dataset.load_dataset = KerasImageDataset.load_dataset

        names = ['mnist', 'fashion_mnist', 'cifar10', 'cifar100']
        for name in names:
            for split in ['train', 'validation']:
                mock_load_data = MagicMock()
                mock_load_data.return_value = (
                    ((np.ones((8, 64, 64)), '{}2'.format(name)),
                     ('{}3'.format(name), '{}4'.format(name)))
                )

                monkeypatch.setattr(
                    ('trainet.datasets.keras_dataset.'
                     'datasets.{}.load_data'.format(name)),
                    mock_load_data
                )

                inputs, targets = keras_dataset.load_dataset(
                    self=keras_dataset, name=name, split=split
                )

                if split == 'train':
                    assert np.all(inputs == 1)
                    assert inputs.ndim == 4
                    assert targets == '{}2'.format(name)
                else:
                    assert inputs.ndim == 1
                    assert np.array_equal(
                        inputs, np.array(['{}3'.format(name)])
                    )
                    assert targets == '{}4'.format(name)
