"""Unit tests for training.pytorch.trainer"""

from unittest.mock import patch, MagicMock

from ktorch.model import Model

from trainet.training.pytorch.trainer import Trainer


class TestTrainer(object):
    """Tests for Trainer"""

    BATCH_SIZE = 3

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
            'trainet.training.pytorch.trainer.validate_config',
            mock_validate_config
        )

        trainer_config = {
            'optimizer': 'Adam', 'loss': 'CrossEntropyLoss',
            'batch_size': self.BATCH_SIZE, 'n_epochs': 2
        }
        dirpath_save = MagicMock()
        trainer = Trainer(
            trainer_config, dirpath_save, gpu_id=1
        )

        assert trainer.dirpath_save == dirpath_save
        assert trainer.optimizer == 'Adam'
        assert trainer.loss == 'CrossEntropyLoss'
        assert trainer.batch_size == self.BATCH_SIZE
        assert trainer.n_epochs == 2
        assert trainer.gpu_id == 1

    def test_train(self, monkeypatch):
        """Test train method"""

        alexnet = MagicMock()
        dataset = MagicMock()

        trainer = MagicMock()
        trainer.n_epochs = 2
        trainer.optimizer = 'Adam'
        trainer.loss = 'CrossEntropyLoss'

        mock_compile = MagicMock()
        monkeypatch.setattr(
            'trainet.training.pytorch.trainer.Model.compile', mock_compile
        )

        mock_cycle = MagicMock()
        mock_cycle_return = MagicMock()
        mock_cycle.return_value = mock_cycle_return
        monkeypatch.setattr(
            'trainet.training.pytorch.trainer.cycle', mock_cycle
        )

        trainer.train = Trainer.train
        with patch.object(Model, 'fit_generator') as fit_fn:
            trainer.train(
                self=trainer,
                train_dataset=dataset, network=alexnet,
                n_steps_per_epoch=1
            )
            assert fit_fn.call_count == 1
            fit_fn.assert_called_with(
                generator=mock_cycle_return, n_steps_per_epoch=1,
                n_epochs=2, validation_data=None, n_validation_steps=None,
                callbacks=None
            )
            assert mock_compile.call_count == 1

            # reset call_count for next assert
            mock_compile.call_count = 0

        with patch.object(Model, 'fit_generator') as fit_fn:
            trainer.train(
                self=trainer,
                train_dataset=dataset, network=alexnet,
                n_steps_per_epoch=1,
                validation_dataset=dataset, n_validation_steps=3
            )
            assert fit_fn.call_count == 1
            fit_fn.assert_called_with(
                generator=mock_cycle_return, n_steps_per_epoch=1,
                n_epochs=2, validation_data=mock_cycle_return,
                n_validation_steps=3, callbacks=None
            )
            assert mock_compile.call_count == 1
