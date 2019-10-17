"""Callbacks for training with Tensorflow"""

import os

import matplotlib.pyplot as plt
import tensorflow.keras.backend as K
import numpy as np
import pandas as pd

from tensorflow.keras.callbacks import (
    Callback,
    EarlyStopping as KerasEarlyStopping,
    ModelCheckpoint as KerasModelCheckpoint,
    ReduceLROnPlateau as KerasReduceLROnPlateau
)


class StateSaverMixIn():
    """MixIn for enabling callbacks to support saving / loading state"""

    def get_state(self):
        """Return the current state

        :return: current state of the callback
        :rtype: dict
        """
        return {}

    def set_state(self, state_dict):
        """Set the state of the callback

        :param state_dict: holds the state to set
        :type state_dict: dict
        """

        for attr_name, attr_value in state_dict.items():
            setattr(self, attr_name, attr_value)


class EarlyStopping(KerasEarlyStopping, StateSaverMixIn):
    """EarlyStopping with saveable state"""

    def __init__(self, *args, **kwargs):
        """Init

        See the tensorflow.keras.callbacks.EarlyStopping class for __init__
        details. The *args and **kwargs are simply passed through to the
        super().__init__().
        """

        super().__init__(*args, **kwargs)

        if self.baseline is not None:
            self.best = self.baseline
        else:
            self.best = np.Inf if self.monitor_op == np.less else -np.Inf

    def on_train_begin(self, logs=None):
        """Run at the beginning of training

        This overwrites the parent class to ensure that `wait` and `best` are
        not overridden, which would happen if the super().on_train_begin() was
        called.

        :param logs: training logs
        :type logs: dict[str: list]
        """

        pass

    def get_state(self):
        """Return the state of the callback

        State of the EarlyStopping callback only depends upon two attributes:
        - self.wait: how many epochs it has been since self.best occured
        - self.best: the current best value of the monitored quantity

        :return: dictionary holding `wait` and `best keys and values
        :rtype: dictionary[str: float]
        """

        return {
            'wait': self.wait,
            'best': self.best
        }


class LRFinder(Callback):
    """Callback for finding the optimal learning rate range

    This callback adjusts the learning rate linearly from `min_lr` to `max_lr`
    during the `n_epochs` of training, recording the loss after each training
    step. At the end of training, it generates and saves plots of the loss
    (both smoothed and unsmoothed) as the learning rate changes.

    Original reference paper: https://arxiv.org/abs/1506.01186
    Reference implementation: https://docs.fast.ai/callbacks.lr_finder.html
    """

    def __init__(self, dirpath_results, n_train_steps_per_epoch,
                 min_lr=1e-5, max_lr=1e-2, n_epochs=1):
        """Init

        :param dirpath_results: directory path to store the results (history
         CSV and plots) in
        :type dirpath_results: str
        :param n_train_steps_per_epoch: number of training steps per epoch
        :type n_train_steps_per_epoch: int
        :param min_lr: minimum learning rate to use during training
        :type min_lr: float
        :param max_lr: maximum learning rate to use during training
        :type max_lr: float
        :param n_epochs: number of epochs to train for
        :type n_epochs: int
        """

        super().__init__()

        self.min_lr = min_lr
        self.max_lr = max_lr

        if not os.path.exists(dirpath_results):
            os.mkdir(dirpath_results)
        self.dirpath_results = dirpath_results

        self.n_total_iterations = n_train_steps_per_epoch * n_epochs
        self.n_min_iterations = int(self.n_total_iterations * 0.10)
        self.averaging_window = int(0.05 * self.n_total_iterations)
        self.iteration = 0
        self.alpha = 0.98

        self.history = {}
        self.df_history = None
        # needs to be set with self.set_model
        self.model = None

    def _calc_learning_rate(self):
        """Calculate the learning rate at a given step

        :return: the new learning rate to use
        :rtype: float
        """

        pct_of_iterations_complete = self.iteration / self.n_total_iterations
        new_learning_rate = (
            self.min_lr + (self.max_lr - self.min_lr) *
            pct_of_iterations_complete
        )

        return new_learning_rate

    def _plot_loss(self, logscale=True):
        """Plot the unsmoothed loss throughout the course of training

        :param logscale: if True, plot using logscale for the x-axis
        :type logscale: bool
        """

        n_passes_to_skip = min(len(self.history['lr']), 10)
        n_passes_to_truncate = int(self.averaging_window * 0.5)

        learning_rates = (
            self.history['lr'][n_passes_to_skip:-n_passes_to_truncate]
        )
        loss_values = (
            self.history['loss'][n_passes_to_skip:-n_passes_to_truncate]
        )

        _, ax = plt.subplots(1, 1)

        ax.plot(learning_rates, loss_values)

        min_loss = np.min(loss_values)
        early_loss_average = np.average(loss_values[10:self.averaging_window])
        # set the y-axis of the plot so that big jumps in the loss don't leave
        # the remainder of the plot un-interpretable; this won't really affect
        # the interpretation of the plot, since a spike off plot still tells us
        # that the learning rate is too high
        plt.ylim(
            min_loss * 0.90,
            max(loss_values[0] * 1.10, early_loss_average * 1.5)
        )
        if logscale:
            ax.set_xscale('log')
        ax.set_xlabel('Learning rate')
        ax.set_ylabel('Loss')

    def _plot_lr(self, logscale=True):
        """Plot the learning rate over the course of training

        :param logscale: if True, plot using logscale for the y-axis
        :type logscale: bool
        """

        plt.plot(self.df_history['iterations'], self.df_history['lr'])
        if logscale:
            plt.yscale('log')
        plt.xlabel('Iteration')
        plt.ylabel('Learning rate')

    def _plot_smoothed_loss(self, logscale=True):
        """Plot the smoothed loss throughout training

        :param bool logscale: if True, plot using logscale for the x-axis
        """

        n_passes_to_skip = min(len(self.history['lr']), 10)
        n_passes_to_truncate = int(self.averaging_window * 0.5)

        learning_rates = (
            self.history['lr'][n_passes_to_skip:-n_passes_to_truncate]
        )
        smoothed_loss = (
            self.history['smoothed_loss'][n_passes_to_skip:-n_passes_to_truncate]
        )

        _, ax = plt.subplots(1, 1)

        ax.plot(learning_rates, smoothed_loss)

        min_loss = np.min(smoothed_loss)
        early_loss_average = (
            np.average(smoothed_loss[10:self.averaging_window])
        )
        # set the y-axis of the plot so that big jumps in the loss don't leave
        # the remainder of the plot un-interpretable; this won't really affect
        # the interpretation of the plot, since a spike off plot still tells us
        # that the learning rate is too high
        plt.ylim(
            min_loss * 0.90,
            max(smoothed_loss[0] * 1.10, early_loss_average * 1.5)
        )
        if logscale:
            ax.set_xscale('log')
        ax.set_xlabel('Learning rate')
        ax.set_ylabel('Smoothed Loss')

    def on_batch_end(self, batch, logs=None):
        """Record previous batch statistics and update the learning rate

        :param batch: index of the current batch
        :type batch: int
        :param logs: training logs
        :type logs: dict[str: list]
        """

        logs = logs or {}
        self.iteration += 1

        self.history.setdefault('lr', []).append(
            K.get_value(self.model.optimizer.lr)
        )
        self.history.setdefault('iterations', []).append(self.iteration)
        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)

        smoothed_loss = self.history.get('smoothed_loss', [0])[-1]
        smoothed_loss = (
            self.alpha * logs['loss'] +
            (1 - self.alpha) * smoothed_loss
        )
        smoothed_loss = smoothed_loss / (1 - self.alpha ** self.iteration)
        self.history.setdefault('smoothed_loss', []).append(smoothed_loss)

        min_loss = np.min(self.history['smoothed_loss'])
        windowed_smoothed_loss = np.average(
            self.history['smoothed_loss'][-self.averaging_window:]
        )
        training_divergent = windowed_smoothed_loss >= (5 * min_loss)
        if self.iteration > self.n_min_iterations and training_divergent:
            self.model.stop_training = True

        new_learning_rate = self._calc_learning_rate()
        K.set_value(self.model.optimizer.lr, new_learning_rate)

    def on_train_begin(self, logs=None):
        """Initialize the learning rate

        :param logs: training logs
        :type logs: dict[str: list]
        """

        logs = logs or {}
        K.set_value(self.model.optimizer.lr, self.min_lr)

    def on_train_end(self, logs=None):
        """Create self.df_history

        :param logs: training logs
        :type logs: dict[str: list]
        """

        self.df_history = pd.DataFrame(self.history)

        fpath_df_history = os.path.join(self.dirpath_results, 'history.csv')
        self.df_history.to_csv(fpath_df_history, index=False)

        for logscale in [True, False]:
            self._plot_loss(logscale=logscale)

            fpath_loss = os.path.join(self.dirpath_results, 'unsmoothed_loss')
            if logscale:
                fpath_loss += '_logscale.png'
            else:
                fpath_loss += '.png'
            plt.savefig(fpath_loss)
            plt.clf()

            self._plot_lr(logscale=logscale)

            fpath_lr = os.path.join(self.dirpath_results, 'lr')
            if logscale:
                fpath_lr += '_logscale.png'
            else:
                fpath_lr = '.png'
            plt.savefig(fpath_lr)
            plt.clf()

            self._plot_smoothed_loss(logscale=logscale)

            fpath_loss_smoothed = os.path.join(
                self.dirpath_results, 'smoothed_loss'
            )
            if logscale:
                fpath_loss_smoothed += '_logscale.png'
            else:
                fpath_loss_smoothed += '.png'
            plt.savefig(fpath_loss_smoothed)
            plt.clf()


class ModelCheckpoint(KerasModelCheckpoint, StateSaverMixIn):
    """ModelCheckpoint with saveable state"""

    def get_state(self):
        """Return the state of the callback

        State of the ModelCheckpoint callback only depends upon one attribute:
        - self.best: the current best value of the monitored quantity

        :return: dictionary holding a `best` key and value
        :rtype: dictionary[str: float]
        """

        return {'best': self.best}


class ReduceLROnPlateau(KerasReduceLROnPlateau, StateSaverMixIn):
    """ReduceLROnPlateau with saveable state"""

    def get_state(self):
        """Return the state of the callback

        State of the ReduceLROnPlateau callback depends on a couple of
        attributes:
        - self.best: the curent best value of the monitored quantity
        - self.wait: how many epochs it has been since self.best occured
        - self.cooldown_counter: if greater than 0, how many iterations it has
          been since the LR was last reduced; else in normal operation mode

        :return: dictionary holding a `best` key and value
        :rtype: dictionary[str: float]
        """

        return {
            'best': self.best,
            'wait': self.wait,
            'cooldown_counter': self.cooldown_counter
        }

    def on_train_begin(self, logs=None):
        """Run on train begin

        This overwrites the parent class to ensure that `wait` and `best` are
        not overridden, which would happen if the super().on_train_begin() was
        called.

        :param logs: training logs
        :type logs: dict[str: list]
        """

        pass


class StateSaver(Callback):
    """Callback for saving TrainingJob state

    When incorporated into a training job, this callack saves a copy of the
    model (using `keras.Model.save()`) along with the state of each callback
    that supports state saving (i.e. has the StateSaverMixIn) at the end of
    each epoch, which allows for resuming model training from the mostly
    recently completed epoch.
    """

    def __init__(self, training_job):
        """Init

        :param training_job: training_job whose state will be saved at the end
         of each epoch
        :type training_job: training.base_training_job.BaseTrainingJob or
         derived subclass
        """

        super().__init__()

        self.training_job = training_job

    def on_train_begin(self, logs=None):
        """Save the initial state at the beginning of training"""

        self.training_job.save_state()

    def on_epoch_end(self, epoch, logs=None):
        """Save state at the end of every epoch

        :param epoch: index of the current epoch
        :type epoch: int
        :param logs: training logs
        :type logs: dict[str: list]
        """

        self.training_job.save_state()
