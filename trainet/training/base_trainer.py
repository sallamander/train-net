"""Class for training a given network on a given dataset"""

from abc import abstractmethod

from trainet.utils.generic_utils import validate_config


class BaseTrainer():
    """BaseTrainer"""

    required_config_keys = {'batch_size', 'loss', 'n_epochs', 'optimizer'}

    def __init__(self, config, dirpath_save, gpu_id=None):
        """Init

        config must contain the following keys:
        - str or dict optimizer: specifies the optimizer to use when training
          the network; if a str, specifies an import path to an
          optimizer to use during training; if a dict, includes an
          'importpath' key holding the importpath to an optimizer to use during
          training and an 'init_params' key holding the params to use when
          initializing the optimizer
        - str or list[str] loss: loss function(s) to use when training the
          network
        - int batch_size: batch size to use during training
        - int n_epochs: number of epochs to train for

        It can additionally contain the following keys:
        - list[float] loss_weights: weights to apply when training a
          multi-output network

        :param config: specifies the configuration of the trainer
        :type config: dict
        :param dirpath_save: directory path to save the model to during
         training
        :type dirpath_save: str
        :param gpu_id: specifies a GPU to use for training
        :type gpu_id: int
        """

        validate_config(config, self.required_config_keys)
        self.config = config

        self.loss = self.config['loss']
        self.loss_weights = self.config.get('loss_weights', None)
        self.batch_size = config['batch_size']
        self.n_epochs = config['n_epochs']

        self.dirpath_save = dirpath_save
        self.gpu_id = gpu_id

        # set in the `train` method
        self.optimizer = None

    def _init_optimizer(self):
        """Initialize the optimizer used to train the network

        :return: initialized optimizer
        :rtype: object
        """

        optimizer_spec = self.config['optimizer']

        if isinstance(optimizer_spec, str):
            optimizer_importpath = optimizer_spec
            init_params = {}
        else:
            optimizer_importpath = optimizer_spec['importpath']
            init_params = optimizer_spec['init_params']

        Optimizer = import_object(optimizer_importpath)
        optimizer = Optimizer(**init_params)
        return optimizer

    @abstractmethod
    def train(self, network, train_dataset, n_steps_per_epoch,
              validation_dataset=None, n_validation_steps=None, metrics=None,
              callbacks=None):
        """Train the network as specified via the __init__ parameters

        :param network: network object to use for training
        :type network: object
        :param train_dataset: dataset that iterates over the training data
         indefinitely
        :type train_dataset: object
        :param n_steps_per_epoch: number of batches to train on in one epoch
        :type n_steps_per_epoch: int
        :param validation_dataset: optional dataset that iterates over the
         validation data indefinitely
        :type validation_dataset: object
        :param n_validation_steps: number of batches to validate on after each
         epoch
        :type n_validation_steps: int
        :param metrics: metrics to be evaluated by the model during training
         and testing
        :type metrics: list[object]
        :param callbacks: callbaks to be used during training
        :type callbacks: list[object]
        """

        raise NotImplementedError
