"""Dataset ops"""

import numpy as np


def apply_transformation(transformation_fn, sample, sample_keys,
                         transformation_fn_kwargs=None):
    """Apply a transformation to certain elements of the provided sample

    :param transformation_fn: function to apply to each of the values located
     at the given `sample_keys` of `sample`
    :type transformation_fn: function
    :param sample: holds the elements to apply the `transformation_fn` to
    :type sample: dict
    :param sample_keys: holds as keys the sample element names to apply the
     transformation to, and as values the transformation_kwarg name to pass the
     sample element in through, e.g. {'image': 'img', 'label': 'target'} will
     result in transformation_fn(img=sample['image'], target=sample['label'])
     plus whatever is specified in transformation_fn_kwargs
    :type sample_keys: dict{str: str}
    :param transformation_fn_kwargs: holds keyword arguments to pass to the
     `transformation_fn`
    :type transformation_fn_kwargs: dict
    :return: `sample` with `transformation_fn` applied to the specified
     elements
    :rtype: dict
    """

    transformation_fn_kwargs = transformation_fn_kwargs or {}
    sample_keys_inverse = {value: key for key, value in sample_keys.items()}

    if len(sample_keys) > 1:
        sample_keys_inverse = {
            value: key for key, value in sample_keys.items()
        }
        for sample_key, transformation_fn_kwarg in sample_keys.items():
            transformation_fn_argument = sample[sample_key]
            transformation_fn_kwargs[transformation_fn_kwarg] = (
                transformation_fn_argument
            )

        transformed_elements = transformation_fn(**transformation_fn_kwargs)

        is_dict = isinstance(transformed_elements, dict)
        has_correct_keys = (
            is_dict and set(transformed_elements) == set(sample_keys_inverse)
        )
        if not (is_dict and has_correct_keys):
            msg = (
                'When applying a transformation to multiple elements of the '
                'sample, the transformation must return a dictionary whose '
                'keys are equal to the kwarg arguments the sample elements '
                'were passed in through, e.g. if applying `my_transform` '
                'to `image` and `label` through the kwargs \'img\' and '
                '\'target\' (i.e. `my_transform(img=image, target=label)`), '
                'the returned value must be a dictionary with keys \'img\' '
                'and \'target\' where the values hold the transformed `image` '
                'and `label`, respectively.'
            )
            raise ValueError(msg)

        it = transformed_elements.items()
        for transformation_fn_kwarg, transformed_element in it:
            sample_key = sample_keys_inverse[transformation_fn_kwarg]
            sample[sample_key] = transformed_element
    else:
        assert len(sample_keys) == 1

        sample_key = list(sample_keys.keys())[0]
        transformation_fn_kwarg = list(sample_keys.values())[0]
        transformation_fn_argument = sample[sample_key]
        transformation_fn_kwargs[transformation_fn_kwarg] = (
            transformation_fn_argument
        )

        transformed_element = transformation_fn(**transformation_fn_kwargs)

        is_dict = isinstance(transformed_element, dict)
        if is_dict:
            has_correct_keys = (
                set(transformed_element) == set(sample_keys_inverse)
            )
            if not has_correct_keys:
                msg = (
                    'When applying a transformation to a single element of '
                    'the sample and returning a dictionary from the '
                    'transformation, the dictionary key must be equal to the '
                    'kwarg argument the sample element was passed in through '
                    ', e.g. if applying `my_transform` to `image` through the '
                    'kwarg \'img\' (i.e. `my_transform(img=image)`), the '
                    'returned value must be a dictionary with an \'img\' key '
                    'and the corresponding value being the transformed '
                    '`image`.'
                )
                raise ValueError(msg)
            sample[sample_key] = transformed_element[transformation_fn_kwarg]
        else:
            sample[sample_key] = transformed_element

    return sample


def per_image_standardization(image):
    """Return the provided `image` with zero mean and unit variance

    This mimics the `tensorflow.image.per_image_standardization`
    implementation, located at
    https://www.tensorflow.org/api_docs/python/tf/image/per_image_standardization.

    :param image: image data to standardize, of shape (height, width,
     n_channels)
    :type image: numpy.ndarray
    :return: `image` standardized to have zero mean and unit variance
    :rtype: numpy.ndarray
    """

    if image.ndim != 3:
        msg = '`image` must have 3 dimensions, but has shape {}'
        raise ValueError(msg.format(image.shape))

    image_mean = image.mean()
    min_std = 1 / np.sqrt(image.size)
    image_std = max(image.std(), min_std)

    image = (image - image_mean) / image_std
    return image
