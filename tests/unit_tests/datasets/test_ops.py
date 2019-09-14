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


class TestApplyTransformation():
    """Tests for `apply_transformation` over different use cases

    General use cases include the following:
    - Applying a function to a single sample element (i.e. using a single
      sample key) that returns a dictionary
    - Applying a function to a single sample element (i.e. using a single
      sample key) that returns the modified sample element
    - Applying a function to multiple sample elements (i.e. using multiple
      sample keys), where it has to return a dictionary
    """

    def test_apply_transformation__image_centering(self, image, label):
        """Test `apply_transformation` with `per_image_standardization`

        This tests the case where a transformation_fn is applied to a single
        sample element (the image) and it returns the modified sample element.
        This test case is a bit more realistic relative to the other test cases
        in this test suite, since it uses an actual function rather than a
        simple mocked one.

        :param image: module wide image object fixture
        :type image: numpy.ndarray
        :param label: module wide label object fixture
        :type label: numpy.ndarray
        """

        sample = {'image': image, 'label': label}
        sample_keys = {'image': 'image'}
        transformation_fn = per_image_standardization

        sample_centered = apply_transformation(
            transformation_fn, sample, sample_keys
        )

        assert sample_centered['image'].shape == sample['image'].shape
        assert np.allclose(sample_centered['image'].mean(), 0, atol=1e-4)
        assert np.allclose(sample_centered['image'].std(), 1, atol=1e-4)
        assert sample_centered['label'] == 1

    def test_apply_transformation__multiple_keys(self, image, label):
        """Test `apply_transformation` with multiple sample keys

        This tests three cases:
        - A transformation is applied to multiple sample elements (both image
          and label elements), and a properly formatted dictionary is returned,
          so no error is thrown
        - A transformation is applied to multiple sample elements (both image
          and label elements), and an improperly formatted dictionary is
          returned, which causes a ValueError
        - A transformation is applied to multiple sample elements (both image
          and label elements), and a non-dictionary is returned, which causes a
          ValueError

        :param image: module wide image object fixture
        :type image: numpy.ndarray
        :param label: module wide label object fixture
        :type label: numpy.ndarray
        """

        sample = {'image': image, 'label': label}
        sample_keys = {'image': 'img', 'label': 'target'}

        def transformation_fn(img, target, return_dict=True,
                              return_correct_format=True):
            """Mock transformation fn

            :param img: image to apply transformation to
            :type img: numpy.ndarray
            :param target: target to apply transformation to
            :type target: numpy.ndarray
            :param return_dict: if True, return a dictionary as the output
             type; if False, return a list (useful for testing that the proper
             ValueError is raised, which it should be if a list is returned)
            :type return_dict: bool
            :param return_correct_format: if True, return a properly formmated
              dictionary; if False, return an improperly formatted dictionary
              (useful for testing that a ValueError is raised); this parameter
              only makes sense if return_dict=True
            :type return_correct_format: bool
            """

            if return_dict:
                if return_correct_format:
                    return {'img': image, 'target': target + 1}
                else:
                    return {'image': image, 'target': target + 1}
            else:
                return img

        transformation_fn_kwargs = {
            'return_dict': True, 'return_correct_format': True
        }
        transformed_sample = apply_transformation(
            transformation_fn, sample, sample_keys, transformation_fn_kwargs
        )

        assert np.array_equal(transformed_sample['image'], image)
        assert transformed_sample['label'] == 2

        with pytest.raises(ValueError):
            transformation_fn_kwargs = {
                'return_dict': True, 'return_correct_format': False
            }
            transformed_sample = apply_transformation(
                transformation_fn, sample, sample_keys,
                transformation_fn_kwargs
            )

        with pytest.raises(ValueError):
            transformation_fn_kwargs = {
                'return_dict': False, 'return_correct_format': False
            }
            transformed_sample = apply_transformation(
                transformation_fn, sample, sample_keys,
                transformation_fn_kwargs
            )

    def test_apply_transformation__single_key(self, image, label):
        """Test `apply_transformation` with a single sample key

        This tests three cases:
        - A transformation is applied to a single sample element (the label),
          and a properly formatted dictionary is returned, so no error is
          thrown
        - A transformation is applied to a single sample element (the label),
          and a improperly formatted dictionary is returned, which causes a
          ValueError
        - A transformation is applied to single sample element (the label)
          and a non-dictionary is returned, so no error is thrown

        :param image: module wide image object fixture
        :type image: numpy.ndarray
        :param label: module wide label object fixture
        :type label: numpy.ndarray
        """

        sample = {'image': image, 'label': label}
        sample_keys = {'label': 'target'}

        def transformation_fn(target, return_dict=True,
                              return_correct_format=True):
            """Mock transformation fn

            :param target: target to apply transformation to
            :type target: numpy.ndarray
            :param return_dict: if True, return a dictionary as the output
             type; if False, return a list (useful for testing that the proper
             ValueError is raised, which it should be if a list is returned)
            :type return_dict: bool
            :param return_correct_format: if True, return a properly formmated
              dictionary; if False, return an improperly formatted dictionary
              (useful for testing that a ValueError is raised); this parameter
              only makes sense if return_dict=True
            :type return_correct_format: bool
            """

            if return_dict:
                if return_correct_format:
                    return {'target': target + 1}
                else:
                    return {'label': target + 1}
            else:
                return target + 1

        transformation_fn_kwargs = {
            'return_dict': True, 'return_correct_format': True
        }
        transformed_sample = apply_transformation(
            transformation_fn, sample, sample_keys, transformation_fn_kwargs
        )
        assert transformed_sample['label'] == 2

        transformation_fn_kwargs = {
            'return_dict': False, 'return_correct_format': True
        }
        transformed_sample = apply_transformation(
            transformation_fn, sample, sample_keys, transformation_fn_kwargs
        )
        assert transformed_sample['label'] == 3

        with pytest.raises(ValueError):
            transformation_fn_kwargs = {
                'return_dict': True, 'return_correct_format': False
            }
            transformed_sample = apply_transformation(
                transformation_fn, sample, sample_keys,
                transformation_fn_kwargs
            )


class TestPerImageStandardization():
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
