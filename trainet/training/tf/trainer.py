"""Trainer for training a Keras model"""

from tensorflow.keras import Model

from trainet.training.base_trainer import BaseTrainer
from trainet.utils.generic_utils import import_object, validate_config


class Trainer(BaseTrainer):
    """Trainer"""

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

    def train(self, network, train_dataset, n_steps_per_epoch,
              validation_dataset=None, n_validation_steps=None, metrics=None,
              callbacks=None):
        """Train the network as specified via the __init__ parameters

        :param network: network object to use for training
        :type network: networks.alexnet_tf.AlexNet
        :param train_dataset: dataset that iterates over the training data
         indefinitely
        :type train_data: tf.data.Dataset
        :param n_steps_per_epoch: number of batches to train on in one epoch
        :type n_steps_per_epoch: int
        :param validation_dataset: optional dataset that iterates over the
         validation data indefinitely
        :type validation_dataset: tf.data.Dataset
        :param n_validation_steps: number of batches to validate on after each
         epoch
        :type n_validation_steps: int
        :param metrics: metrics to be evaluated by the model during training
         and testing
        :type metrics: list[object]
        :param callbacks: callbacks to be used during training
        :type callbacks: list[object]
        """
        
        self.optimizer = self._init_optimizer()

        inputs, outputs = network.build()
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(
            optimizer=self.optimizer, loss=self.loss, metrics=metrics
        )

        model.fit(
            x=train_dataset,
            steps_per_epoch=n_steps_per_epoch,
            epochs=self.n_epochs,
            verbose=True,
            validation_data=validation_dataset,
            validation_steps=n_validation_steps,
            callbacks=callbacks
        )
