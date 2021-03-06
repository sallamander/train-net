"""Unit tests for dataset.toy_image_dataset"""

from unittest.mock import MagicMock

import numpy as np
import pytest
from scipy.ndimage.measurements import center_of_mass

from trainet.datasets.toy_image_dataset import (
    generate_ellipse_coordinates, generate_rectangle_coordinates,
    generate_triangle_coordinates, ToyImageDataset
)

TEST_CASES = [
    {'image_shape': (64, 64), 'centerpoint': (32, 32),
     'size_bin': 'small'},
    {'image_shape': (96, 96), 'centerpoint': (48, 48),
     'size_bin': 'medium'},
    {'image_shape': (128, 128), 'centerpoint': (64, 64),
     'size_bin': 'large'},
]


def check_centerpoint(coordinates, image_shape, expected_centerpoint, atol=1):
    """Assert that the centerpoint of coordinates is as expected

    :param coordinates: coordinates to check
    :type coordinates: tuple(numpy.ndarray)
    :param image_shape: holds the image shape that the coordinates were
     generated for
    :type image_shape: tuple(int)
    :param expected_centerpoint: holds the expected centerpoint
    :type expected_centerpoint: tuple(float)
    :param atol: tolerance to allow when checking the centerpoint
    :type atol: int
    """

    image = np.zeros(image_shape)
    image[coordinates] = 1
    centerpoint = center_of_mass(image)
    assert np.allclose(centerpoint, expected_centerpoint, atol=atol)


def check_size(coordinates, size_bin, image_shape):
    """Assert that coordinates take up the expected fraction of `image_shape`

    :param coordinates: coordinates to check
    :type coordinates: tuple(numpy.ndarray)
    :param size_bin: denotes the size of the shape that the coordinates
     represent, one of 'small', 'medium', or 'large'
    :type size_bin: str
    :param image_shape: holds the image shape that the coordinates were
     generated for
    :type image_shape: tuple(int)
    """

    size_y = (
        float((max(coordinates[0]) - min(coordinates[0]))) / image_shape[0]
    )
    size_x = (
        float((max(coordinates[1]) - min(coordinates[1]))) / image_shape[1]
    )

    if size_bin == 'small':
        min_size = 0.015
        max_size = 0.15
    elif size_bin == 'medium':
        min_size = 0.10
        max_size = 0.40
    elif size_bin == 'large':
        min_size = 0.25
        max_size = 0.70

    assert min_size <= size_y <= max_size
    assert min_size <= size_x <= max_size


def test_generate_ellipse_coordinates():
    """Test generate_ellipse_coordinates method

    This acts somewhat like a smoke test, and tests a couple of things:
    - The returned ellipse takes up roughly the correct portion of the overall
      image
    - The centerpoint of the returned ellipse is in the right place
    """

    for test_case in TEST_CASES:
        ellipse_coordinates = generate_ellipse_coordinates(**test_case)

        check_size(
            ellipse_coordinates, test_case['size_bin'],
            test_case['image_shape']
        )
        check_centerpoint(
            ellipse_coordinates, test_case['image_shape'],
            test_case['centerpoint']
        )


def test_generate_rectangle_coordinates():
    """Test generate_rectangle_coordinates method

    This acts somewhat like a smoke test, and tests a couple of things:
    - The returned rectangle takes up roughly the correct portion of the
      overall image
    - The centerpoint of the returned rectangle is in the right place
    """

    for test_case in TEST_CASES:
        rectangle_coordinates = generate_rectangle_coordinates(**test_case)

        check_size(
            rectangle_coordinates, test_case['size_bin'],
            test_case['image_shape']
        )
        check_centerpoint(
            rectangle_coordinates, test_case['image_shape'],
            test_case['centerpoint']
        )


def test_generate_triangle_coordinates():
    """Test generate_triangle_coordinates method

    This acts somewhat like a smoke test, and tests a couple of things:
    - The returned triangle takes up roughly the correct portion of the
      overall image
    - The centerpoint of the returned triangle is in the right place
    """

    for test_case in TEST_CASES:
        triangle_coordinates = generate_triangle_coordinates(**test_case)

        check_size(
            triangle_coordinates, test_case['size_bin'],
            test_case['image_shape']
        )
        check_centerpoint(
            triangle_coordinates, test_case['image_shape'],
            test_case['centerpoint'], atol=15
        )


class TestToyImageDataset():
    """Tests for ToyImageDataset"""

    @pytest.fixture(scope='class')
    def dataset_config(self):
        """dataset_config object fixture

        :return: dataset_config to be used to instantiate a ToyImageDataset
        :rtype: dict
        """

        return {
            'height': 64, 'width': 64, 'n_classes': 5,
            'object_colors': ['yellow', 'orange'],
            'object_shapes': ['ellipse'],
        }

    def test_init(self, dataset_config, monkeypatch):
        """Test __init__ method

        This tests that all attributes are set correctly in the __init__.

        :param dataset_config: dataset_config object fixture
        :type dataset_config: dict
        :param monkeypatch: monkeypatch object
        :type monkeypatch: _pytest.monkeypatch.MonkeyPatch
        """

        mock_validate_config = MagicMock()
        monkeypatch.setattr(
            'trainet.datasets.toy_image_dataset.validate_config',
            mock_validate_config
        )

        dataset = ToyImageDataset(dataset_config)
        assert dataset.height == 64
        assert dataset.width == 64
        assert dataset.n_classes == 5
        assert dataset.object_colors == ['orange', 'yellow']
        assert dataset.object_shapes == ['ellipse']
        assert dataset.object_sizes == ['small', 'medium', 'large']
        assert len(dataset.object_spec_options) == 5
        assert dataset.size == 5
        assert len(dataset) == 5

    def test_init__errors(self, dataset_config, monkeypatch):
        """Test __init__ method

        This tests that if a n invalid object shape, size, or color is
        specified, a ValueError is raised.

        :param dataset_config: dataset_config object fixture
        :type: dict
        :param monkeypatch: monkeypatch object
        :type monkeypatch: _pytest.monkeypatch.MonkeyPatch
        """

        mock_validate_config = MagicMock()
        monkeypatch.setattr(
            'trainet.datasets.toy_image_dataset.validate_config',
            mock_validate_config
        )

        with pytest.raises(ValueError):
            dataset_config = dataset_config.copy()
            dataset_config['object_shapes'] = ['taco']
            ToyImageDataset(dataset_config)

        with pytest.raises(ValueError):
            dataset_config = dataset_config.copy()
            dataset_config['object_sizes'] = ['gigantic']
            ToyImageDataset(dataset_config)

        with pytest.raises(ValueError):
            dataset_config = dataset_config.copy()
            dataset_config['object_colors'] = ['sky-blue']
            ToyImageDataset(dataset_config)

    def test_getitem(self):
        """Test __getitem__ method"""

        mock_dataset = MagicMock()
        mock_dataset.height = 64
        mock_dataset.width = 64
        mock_get_object_coordinates = MagicMock()
        mock_get_object_coordinates.return_value = ([0], [1])

        mock_dataset._get_object_coordinates = mock_get_object_coordinates
        mock_dataset.object_spec_options = {
            0: ('blue', 'triangle', 'small')
        }

        mock_dataset.__getitem__ = ToyImageDataset.__getitem__
        sample = mock_dataset[2]
        image, label = sample['image'], sample['label']

        assert image.shape == (64, 64, 3)
        assert image[0, 1, 0] == 0
        assert image[0, 1, 1] == 0
        assert image[0, 1, 2] == 1
        assert label == 0

    def test_get_object_coordinates(self, monkeypatch):
        """Test get_object_coordinates method

        :param monkeypatch: monkeypatch object
        :type monkeypatch: _pytest.monkeypatch.MonkeyPatch
        """

        mock_dataset = MagicMock()
        mock_dataset.height = 64
        mock_dataset.width = 64

        mock_generate_ellipse_coordinates = MagicMock()
        mock_generate_ellipse_coordinates.return_value = (
            [4, 5, 7, 65, 66, 12], [7, 8, 9, 12, 68, 7]
        )
        monkeypatch.setattr(
            'trainet.datasets.toy_image_dataset.generate_ellipse_coordinates',
            mock_generate_ellipse_coordinates
        )

        mock_dataset._get_object_coordinates = (
            ToyImageDataset._get_object_coordinates
        )
        object_coordinates = mock_dataset._get_object_coordinates(
            self=mock_dataset, object_shape='ellipse', object_size='small'
        )

        assert np.array_equal(
            object_coordinates[0], [4, 5, 7, 63, 63, 12]
        )
        assert np.array_equal(
            object_coordinates[1], [7, 8, 9, 12, 63, 7]
        )
        assert mock_generate_ellipse_coordinates.call_count == 1
