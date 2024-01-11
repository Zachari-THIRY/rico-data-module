import torch
import numpy as np
from torchtyping import TensorType
from typing import Tuple
from rico_data_module import RICODataset, trstr

def get_data(r_params, s_params, r, ignore_warnings=False) -> Tuple[RICODataset, RICODataset]:
    # Returns the train and test RicoDatasets
    real = RICODataset(r_params)
    synth = RICODataset(s_params)

    train, test = trstr(real, synth, r, ignore_warnings) # NB only the r is retrieved from params (so far)
    return train, test


def get_real_synth(experiment:str, synthetiser:str, shuffle:bool=False
                    ) -> Tuple[np.array, np.array]:
    real_path = f'datasets/{experiment}/{experiment}_FULL.tsv'
    synth_path = f'datasets/{experiment}/{experiment}_{synthetiser}.tsv'

    real = np.loadtxt(real_path, delimiter='\t', dtype=float, usecols = range(1, 29))
    synth = np.loadtxt(synth_path, delimiter='\t', dtype=float, usecols = range(1, 29))

    if shuffle:
        np.random.shuffle(real)
        np.random.shuffle(synth)

    return real, synth


def get_train_test(
    r:float, 
    synthetiser:str, 
    real_in_train:float, 
    experiment:str, 
    shuffle:bool
    ) -> Tuple[np.array, np.array]:
    """
    r: float
        Ratio of real samples in the training set
    synthetiser: str
        Name of the synthetiser
    real_in_train: float
        Ratio of real samples in the training set (1 -> TRTR, 0 -> TSTR)
    experiment: str
        Name of the experiment
    shuffle: bool
        If True, shuffle the real and synthetic data
    """

    real, synth = get_real_synth(experiment, synthetiser, shuffle)

    if r ==1 :      # Case TSTR
        return torch.from_numpy(synth).type(torch.Tensor), torch.from_numpy(real).type(torch.Tensor)
    else:           # Case TRSTR
        n_real_in_train = int(len(real) * real_in_train)
        n_synth_in_train = min(int(r * n_real_in_train /(1-r)), len(synth))

        train = np.concatenate((real[:n_real_in_train], synth[:n_synth_in_train]))
        test = real[n_real_in_train:]
        return torch.from_numpy(train).type(torch.Tensor), torch.from_numpy(test).type(torch.Tensor)

def build_data_pipeline(synthetiser:str,
                        r:float,
                        real_in_train:float, 
                        experiment:str, 
                        shuffle:bool,
                        target_len:int, 
                        batch_first:bool=True) -> Tuple[TensorType["B,L,C"], TensorType["B,L,C"], TensorType["B,L-1,C"], TensorType["B,1,C"]]:
    """
    Takes a train and test set and splits the train set into train, validation and test X and y sets.
    Divides a train_set into train & val (N,L,C), X and y
    Prepares train, val and test sets accordingly (L,N,C) or (N,L,C) if batch_first

    Parameters
    ----------
    synthetiser : str
        Name of the synthetiser.
    r: float
        Ratio of real samples in the training set
    real_in_train : float
        Ratio of real samples in the training set.
    experiment : str
        Name of the experiment.
    shuffle : bool
        If True, shuffle the real and synthetic data.
    target_len : int
        Length of the target sequence.
    batch_first : bool, optional
        Whether to return the data in (L,N,C) or (N,L,C) format, by default True.

    Returns
    -------
    tuple
        A tuple containing (train_X, train_y), (test_X, test_y).
        train_X : TensorType["B,L,C"]
            The input training sequences.
        train_y : TensorType["B,L-1,C"]
            The target training sequences.
        test_X : TensorType["B,L,C"]
            The input test sequences.
        test_y : TensorType["B,1,C"]
            The target test sequences.
    """

    train, test = get_train_test(r=r, 
                                 synthetiser = synthetiser, 
                                 real_in_train = real_in_train, 
                                 experiment = experiment,
                                 shuffle=shuffle) # Shape (B,L)

    train, test = train.unsqueeze(2), test.unsqueeze(2) # Shape (B,L,1)

    if target_len == 0:
        if not batch_first:
            train, test = train.permute(1,0,2), test.permute(1,0,2)
        return train, test
    

    train_X, train_y = train[:,:-target_len], train[:,-target_len:]
    test_X, test_y = test[:,:-target_len], test[:,-target_len:]


    if not batch_first:
        train_X, train_y = train_X.permute(1,0,2), train_y.permute(1,0,2)
        test_X, test_y = test_X.permute(1,0,2), test_y.permute(1,0,2)

    return (train_X, train_y), (test_X, test_y)