"""Unit tests for training.tf.trainer"""

from unittest.mock import patch, MagicMock

from tensorflow.keras import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.optimizers import Adam

from trainet.training.tf.trainer import Trainer


class TestTrainer(object):
    """Tests for Trainer"""

    BATCH_SIZE = 3
    HEIGHT = 227
    WIDTH = 227
    NUM_CHANNELS = 3

    def get_alexnet(self):
        """Return a mock networks.alexnet.AlexNet object

        :return: alexnet model to use during training
        :rtype: unittest.mock.MagicMock
        """

        def mock_forward():
            """Return mock `inputs` and `outputs`"""

            inputs = Input(shape=(self.HEIGHT, self.WIDTH, self.NUM_CHANNELS))
            outputs = inputs
            return inputs, outputs

        alexnet = MagicMock()
        alexnet.forward = mock_forward
        return alexnet

    def test_init(self, monkeypatch):
        """Test __init__ method

        This tests that all attributes are set correctly in the __init__.

        :param monkeypatch: monkeypatch object
        :type monkeypatch: _pytest.monkeypatch.MonkeyPatch
        """

        def mock_validate_config(config, required_keys):
            """Mock validate_config to pass"""
            pass
        monkeypatch.setattr(
            'trainet.training.tf.trainer.validate_config',
            mock_validate_config
        )

        trainer_config = {
            'optimizer': 'adam', 'loss': 'categorical_crossentropy',
            'batch_size': self.BATCH_SIZE, 'n_epochs': 2
        }
        dirpath_save = MagicMock()
        trainer = Trainer(trainer_config, dirpath_save)

        assert trainer.dirpath_save == dirpath_save
        assert trainer.optimizer is None
        assert trainer.loss == 'categorical_crossentropy'
        assert trainer.batch_size == self.BATCH_SIZE
        assert trainer.n_epochs == 2

    def test_train(self):
        """Test train method"""

        alexnet = self.get_alexnet()
        dataset = MagicMock()
        trainer = MagicMock()
        trainer.n_epochs = 2
        trainer._init_optimizer = MagicMock()
        trainer._init_optimizer.return_value = Adam()
        trainer.loss = 'categorical_crossentropy'

        trainer.train = Trainer.train
        with patch.object(Model, 'fit') as fit_fn:
            trainer.train(
                self=trainer,
                train_dataset=dataset, network=alexnet,
                n_steps_per_epoch=1
            )
            assert fit_fn.call_count == 1
            fit_fn.assert_called_with(
                x=dataset, steps_per_epoch=1,
                epochs=2, verbose=True, validation_data=None,
                validation_steps=None, callbacks=None
            )

        with patch.object(Model, 'fit') as fit_fn:
            trainer.train(
                self=trainer, train_dataset=dataset,
                network=alexnet, validation_dataset=dataset,
                n_steps_per_epoch=45, n_validation_steps=2
            )
            assert fit_fn.call_count == 1
            fit_fn.assert_called_with(
                x=dataset, steps_per_epoch=45,
                epochs=2, verbose=True, validation_data=dataset,
                validation_steps=2, callbacks=None
            )
