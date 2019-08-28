"""Trainer for training a pytorch model"""

from ktorch.model import Model

from trainet.training.base_trainer import BaseTrainer
from trainet.utils.generic_utils import cycle, import_object, validate_config


class Trainer(BaseTrainer):
    """Trainer"""

    def _init_optimizer(self, network):
        """Initialize the optimizer used to train the network

        :return: initialized optimizer
        :rtype: object
        """

        optimizer_spec = self.config['optimizer']

        if isinstance(optimizer_spec, str):
            optimizer_importpath = optimizer_spec
            init_params = {'params': network.parameters()}
        else:
            optimizer_importpath = optimizer_spec['importpath']
            init_params = optimizer_spec['init_params'].copy()
            init_params['params'] = network.parameters()

        Optimizer = import_object(optimizer_importpath)
        optimizer = Optimizer(**init_params)
        return optimizer

    def train(self, network, train_dataset, n_steps_per_epoch,
              validation_dataset=None, n_validation_steps=None, metrics=None,
              callbacks=None):
        """Train the network as specified via the __init__ parameters

        :param network: network object to use for training
        :type network: networks.alexnet_pytorch.AlexNet
        :param train_dataset: dataset that iterates over the training data
         indefinitely
        :type train_data: torch.utils.data.DataLoader
        :param n_steps_per_epoch: number of batches to train on in one epoch
        :type n_steps_per_epoch: int
        :param validation_dataset: optional dataset that iterates over the
         validation data indefinitely
        :type validation_dataset: torch.utils.data.DataLoader
        :param n_validation_steps: number of batches to validate on after each
         epoch
        :type n_validation_steps: int
        :param metrics: metrics to be evaluated by the model during training
         and testing
        :type metrics: list[object]
        :param callbacks: callbacks to be used during training
        :type callbacks: list[object]
        """

        self.optimizer = self._init_optimizer(network)

        model = Model(network, self.gpu_id)
        model.compile(
            optimizer=self.optimizer, loss=self.loss, metrics=metrics
        )

        if validation_dataset:
            validation_dataset = cycle(validation_dataset)

        model.fit_generator(
            generator=cycle(train_dataset),
            n_steps_per_epoch=n_steps_per_epoch,
            n_epochs=self.n_epochs,
            validation_data=validation_dataset,
            n_validation_steps=n_validation_steps,
            callbacks=callbacks
        )
