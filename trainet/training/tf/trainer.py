"""Trainer for training a Keras model"""

from tensorflow.keras import Model

from trainet.training.base_trainer import BaseTrainer
from trainet.utils.generic_utils import import_object, validate_config


class Trainer(BaseTrainer):
    """Trainer"""

    def init_optimizer(self):
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

    def load_model(self, network, metrics, optimizer):
        """Load model

        :param network: network object to use for training
        :type network: networks.alexnet_tf.AlexNet
        :param metrics: metrics to be evaluated by the model during training
         and testing
        :type metrics: list[object]
        :return: compiled model to train
        :rtype: keras.Model
        """

        inputs, outputs = network.build()
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(
            optimizer=optimizer, loss=self.loss, metrics=metrics
        )

        return model

    def set_model(self, model):
        """Set model state"""
        
        self.model = model

    def train(self, train_dataset, n_steps_per_epoch,
              validation_dataset=None, n_validation_steps=None, callbacks=None,
              initial_epoch=0):
        """Train self.model as specified via the __init__ parameters

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
        :param callbacks: callbacks to be used during training
        :type callbacks: list[object]
        :param initial_epoch: epoch at which to start training
        :type initial_epoch: int
        """

        msg = (
            '\'self.model is None\', but it must be set before calling '
            '\'train\'. Use the `set_model` method to set it.'
        )
        assert self.model is not None, msg

        self.model.fit(
            x=train_dataset,
            steps_per_epoch=n_steps_per_epoch,
            epochs=self.n_epochs,
            verbose=True,
            validation_data=validation_dataset,
            validation_steps=n_validation_steps,
            callbacks=callbacks,
            initial_epoch=initial_epoch
        )
