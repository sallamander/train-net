"""Unit tests for datasets.ops"""

import numpy as np
import pytest

from trainet.datasets.ops import (
    apply_transformation, per_image_standardization
)


@pytest.fixture(scope='module')
def image():
    """Return a tensorflow.Tensor holding an image object fixture

    :return: tensorflow.Tensor holding an image to use in tests
    :rtype: tensorflow.Tensor
    """

    height, width = np.random.randint(128, 600, 2)
    num_channels = 3
    image = np.random.random((height, width, num_channels))

    return image


@pytest.fixture(scope='module')
def label():
    """Return a tensorflow.Tensor holding an label object fixture

    :return: tensorflow.Tensor holding a label to use in tests
    :rtype: tensorflow.Tensor
    """

    label = 1
    return label


class TestApplyTransformation(object):
    """Tests for `apply_transformation` over different use cases"""

    def test_apply_transformation__image_centering(self, image, label):
        """Test `apply_transformation` with `per_image_standardization`

        This only tests the centering of the 'image' key in the `sample` below,
        but 'label' is still included to simulate a more realistic scenario
        where the `sample` has both an 'image' and 'label' key.

        :param image: module wide image object fixture
        :type image: numpy.ndarray
        :param label: module wide label object fixture
        :type label: numpy.ndarray
        """

        sample = {'image': image, 'label': label}
        sample_keys = {'image'}
        transformation_fn = per_image_standardization

        sample_centered = apply_transformation(
            transformation_fn, sample, sample_keys
        )

        assert sample_centered['image'].shape == sample['image'].shape
        assert np.allclose(sample_centered['image'].mean(), 0, atol=1e-4)
        assert np.allclose(sample_centered['image'].std(), 1, atol=1e-4)
        assert sample_centered['label'] == 1


class TestPerImageStandardization(object):
    """Test `per_image_standardization`"""

    def test_per_image_standardization(self):
        """Test `per_image_standardization` on a non-uniform image"""

        image = np.random.random((227, 227, 3))
        image_standardized = per_image_standardization(image)

        assert np.allclose(image_standardized.mean(), 0)
        assert np.allclose(image_standardized.std(), 1)

        with pytest.raises(ValueError):
            image = np.random.random((1, 2, 3, 4))
            image_standardized = per_image_standardization(image)

    def test_per_image_standardization__uniform(self):
        """Test `per_image_standardization` on a uniform image

        The main point of this test is to ensure that there is no division by
        zero because of the uniformity of the image. In this case, we expect
        that the standard deviation of the pixel values will be 0, but that the
        resulting mean will still also be 0.
        """

        image = np.ones((227, 227, 3))
        image_standardized = per_image_standardization(image)

        assert np.allclose(image_standardized.mean(), 0)
        assert np.allclose(image_standardized.std(), 0)
