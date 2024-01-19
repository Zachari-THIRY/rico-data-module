from numpy import dtype
import torch
import pandas as pd

from sklearn.preprocessing import StandardScaler

from typing import Tuple, List
from torchtyping import TensorType
from .types import DatasetParams


class BaseDataset():
    def __init__(self, data: TensorType["N", "L", "C"] = None) -> None:
        self.data: torch.Tensor["N", "L", "C"] = data
        self.name: str = None
        self.shape: torch.Size = self.size()

    def load(self, params:DatasetParams):
        """
        Load the dataset
        """
        # Sets self.data, and self.name
        raise NotImplementedError
    
    def standardize(self):
        """
        Standardize the dataset
        """
        self.assert_initialized()
        condition = self.data.dim() == 2 or (self.data.dim() == 3 and self.data.shape[-1] == 1)
        assert condition, "Dataset must be 2D or 3D with 1 channel"
        if self.data.dim() == 3:
            self.data = self.data.squeeze(-1)
        self.scaler = StandardScaler()
        self.data = torch.tensor(self.scaler.fit_transform(self.data), dtype=torch.float32)
        self.data = self.data.unsqueeze(-1)
    
    def get_scaler(self):
        """
        Return the scaler used to standardize the dataset
        """
        self.assert_initialized()
        return self.scaler
    
    def merge(self, dataset:'BaseDataset', name:str):
        """
        Merge the dataset with another dataset
        """
        self.assert_initialized()
        dataset.assert_initialized()

        new = BaseDataset()

        new.name = name
        new.data = torch.cat((self.data, dataset.data), dim=0)
        new.ori_indices = torch.arange(len(self))
        new.extra_indices = torch.arange(len(self), len(self) + len(dataset))

        new.update_size()

        return new
    
    def split(self, ratio:float, names:List[str], shuffle = False) -> Tuple['BaseDataset', 'BaseDataset']:
        """
        Split the dataset into two datasets
        """
        self.assert_initialized()
        split = int(len(self) * ratio)
        one, two = BaseDataset(), BaseDataset()
        one.name = self.name + "_1" if not names else names[0]
        two.name = self.name + "_2" if not names else names[1]

        if shuffle:
            data = torch.clone(self.data)
            idx = torch.randperm(len(data))
            data = data[idx]
        else:
            data = self.data

        one.data = self.data[:split]
        two.data = self.data[split:]

        one.update_size()
        two.update_size()

        return one, two
    
    def update_size(self):
        self.shape = self.size()

    def size(self):
        try:
            return self.data.shape
        except AttributeError:
            return 0
    
    def __len__(self):
        """
        Return the length of the dataset
        """
        self.assert_initialized()

        return len(self.data)
            
    def __str__(self) -> str:
        self.assert_initialized()

        str = f"Name: {self.name} " + "\n" + \
        f"Shape : {self.data.size()}"
        return str
    
    def __getitem__(self, index):
        """
        Get an item from the dataset at the specified index
        Parameters
        ----------
        index : int
            The index of the item to retrieve
        
        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            The X and y tensors at the specified index
        """
        return self.data[index]
    
    def __iter__(self):
        for i in range(len(self)):
            yield self[i]
    
    def __add__(self, dataset:'BaseDataset'):
        """
        Merge the dataset with another dataset
        """
        return self.merge(dataset, self.name + "+" + dataset.name)

    def to(self, to:str, _dtype:type=torch.float32) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Exports the dataset to a specific format
        Parameters
        ----------
        to : str (torch, numpy, pandas)
            The format to export the dataset
        """
        self.assert_initialized()
        exp = None
        match to:
            case "torch":
                exp = self.data.to(dtype=_dtype)
            case "numpy":
                exp = self.data.numpy().astype(_dtype)
            case "pandas":
                assert self.data.shape[-1] == 1, "Only single channel data can be converted to pandas"
                exp = pd.DataFrame(self.data.squeeze().numpy().astype(_dtype))
            case _:
                raise ValueError(f"Unknown format : {to}")
        
        return exp
            
    def assert_initialized(self):
        assert self.name is not None, "Please load dataset first"
        assert self.data is not None, "Please load dataset first"

def trstr(real:BaseDataset, synth: BaseDataset, r:float, ignore_warnings:bool=False) -> Tuple[BaseDataset, BaseDataset]:
    """
    Split the dataset into train and test sets
    Test contains 50% of the real dataset
    Parameters
    ----------
    real : BaseDataset
        The real dataset
    synth : BaseDataset
        The synthetic dataset
    r : float
        The ratio of synth in the train set"""
    assert 0<=r<=1, "r must be between 0 and 1"

    match r:
        case 0: # TRTR
            train_r, test_r = real.split(0.5, ["train_r", "test_r"]) # train, test
            return train_r, test_r
        case 1: # TSTR
            return synth, real # train, test
        case _: # TRSTR
            train_r, test_r = real.split(0.5, ["train_r", "test_r"])
            
            n_synths = len(train_r) * r / (1 - r)
            split_ratio = n_synths / len(synth)
            if split_ratio > 1 and not ignore_warnings:
                print(f"\033[91m\033[1mWarning:\033[0m split_ratio is greater than 1, "+ \
                  f"Increase the size of the synth dataset to avoid this warning : {split_ratio:.2f}" + \
                    "\n\t\033[92mDefaulting to 1\033[0m")
                split_ratio = 1
            train_s, _ = synth.split(split_ratio, ["train_s", "test_s"])
            train = train_s.merge(train_r, "train")
            return train, test_r