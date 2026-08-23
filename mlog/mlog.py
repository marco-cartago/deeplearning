import torch.nn as nn
import pickle
import os
import time

from mamba import MambaConfig

class ModelLog(object):

    """
    Will save to an external file a copy of the models of the trained model and 
    various statistics of the training run.

    Class for logging training run data and comprehensive training times.
    the fields of this class begin mainly with `e_` and `s_` containing both
    lists of value of a model statistic during training. 
    
    For example, `e_loss` will be a vector containing the epoch losses of the 
    training run. `s_loss` will record the loss for each gradient descent step.
    """

    def __init__(self, model_config: MambaConfig):

        # Model specs
        self.model_config = model_config

        # Stepsize 
        self.e_stepsize = []
        self.s_stepsize = []

        # Loss
        self.e_train_loss = []
        self.s_train_loss = []
        self.e_validation_loss = []

        # Time elapsed per step
        self.e_time_elapsed = []
        self.s_time_elapsed = []

        # Generic
        self.total_time: int | None = None
        self.n_epochs: int | None = None
        self.n_steps: int | None = None
        self.batch_size: int | None = None

        self.dump_timestamp = 0


    def dump_to_file(self, log_folder):

        mc = self.model_config
        model_name = "mamba-D{d}-E{e:.1f}-N{n}-d{t}"
        model_name = model_name.format(
            d = mc.d_model,
            e = mc.expand_factor,
            n = mc.d_state,
            t = mc.n_layers,
        )
        t = time.time_ns()
        self.dump_timestamp = t

        path = os.path.join(
            log_folder, 
            f"{model_name}" +
            f"{t}d.pkl"
        )

        with open(path, "wb") as file:
            pickle.dump(self, file)
