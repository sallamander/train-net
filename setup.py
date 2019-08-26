"""Setup file"""

from setuptools import find_packages, setup

setup(
    name='train-net',
    version='0.1',
    description=(
        'A repository providing a YML-based training paradigm for neural '
        'networks.'
    ),
    long_description=(
        'train-net aims to consolidate a lot of the boilerplate code that '
        'gets written to train neural networks, allowing users to write '
        'minimal amounts of code to get up and running training networks. '
    ),
    author='Sean Sall',
    author_email='ssall@alumni.nd.edu',
    url='https://github.com/sallamander/train-net',
    download_url=(
        'https://github.com/sallamander/train-net/archive/v0.1-alpha.tar.gz'
    ),
    license='MIT',
    install_requires=[
        'scikit-image>=0.15.0',
        'pandas>=0.25.1',
        'pyaml>=19.4.1',
        'ktorch>=0.5.1'
    ],
    extras_require={
        'tests': [
            'pytest',
            'pytest-pep8'
        ]
    },
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],
    packages=find_packages()
)
