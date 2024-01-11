import warnings

import torch
import pandas as pd
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset
from typing import Optional, List, Tuple
import os

def deprecated_class(cls):
    orig_init = cls.__init__
    
    def new_init(self, *args, **kwargs):
        warnings.warn(cls.__name__ + " is a deprecated class", category=DeprecationWarning)
        orig_init(self, *args, **kwargs)
    
    cls.__init__ = new_init
    return cls

class RICORawDataset(Dataset):
    """
    A PyTorch dataset class for handling RICO raw data.

    Arguments
    ---------
        data (pd.DataFrame): The input data as a pandas DataFrame.
        ori_seq_len (int): The length of the original sequence.
        tgt_seq_len (Optional[int]): The length of the target sequence when using sliding windows . If not provided, it defaults to ori_seq_len.
        stride (int): The stride value for creating sliding windows. Defaults to 1.
        channels (List[str]): The list of column names to be used as channels. If not provided, all columns are used.
        sampling_rate (int): The sampling rate for downsampling the data. Defaults to 1.
        standardize (bool): Flag indicating whether to standardize the data. Defaults to False.

    Attributes
    ----------
        ori_seq_len (int): The length of the original sequence.
        tgt_seq_len (int): The length of the target sequence when using sliding windows.
        stride (int): The stride value for creating sliding windows.
        data (torch.Tensor): The input data as a torch.Tensor.
        num_series (int): The number of series in the data.
        series_length (int): The length of each series.
        scaler (StandardScaler): The scaler object used for standardization.
        channels (List[str]): The list of column names used as channels.
        n_channels (int): The number of channels.
        sampling_rate (int): The sampling rate for downsampling the data.

    Methods
    -------
        __len__(): Returns the length of the dataset.
        __getitem__(idx): Returns the item at the given index.
        __getserie__(idx): Returns the series at the given index.
        export(dir, filename, split): Exports the dataset to a CSV file.

    """

    def __init__(self, data: pd.DataFrame, ori_seq_len:int, tgt_seq_len: Optional[int] = None, stride: int = 1, channels:List[str] =[], sampling_rate:int=1, standardize: bool = False):
        self.ori_seq_len = ori_seq_len
        self.tgt_seq_len = tgt_seq_len if tgt_seq_len else ori_seq_len

        self.stride = stride
        self.data = data[channels] if channels else data
        self.num_series, self.series_length = data.shape
        self.scaler = StandardScaler()
        self.channels = self.data.columns
        self.n_channels = len(self.data.columns)
        self.sampling_rate=sampling_rate

        if standardize: self.data = torch.tensor(self.scaler.fit_transform(self.data.values), dtype=torch.float32)
        else: self.data = torch.tensor(self.data.values, dtype=torch.float32)

        assert self.ori_seq_len <= len(data), ValueError(f'Seq_len {self.ori_seq_len} should be lower than series length {len(data)}')
        self._len = int(len(data) / self.ori_seq_len) * ((self.ori_seq_len - self.tgt_seq_len )//self.stride + 1)

    def __len__(self):
        return self._len
    
    def __getitem__(self, idx):
        series_index = idx // ((self.ori_seq_len - self.tgt_seq_len )//self.stride + 1) 
        pos_in_series = idx - series_index*((self.ori_seq_len - self.tgt_seq_len)//self.stride + 1)

        return self.__getserie__(series_index)[pos_in_series:pos_in_series + self.tgt_seq_len]

    def __getserie__(self, idx):
        if idx >= len(self):
            raise IndexError(f"Index {idx} is out of bounds for length {len(self)}. Please ensure idx < length.")
        start = self.ori_seq_len * idx
        end = start + self.ori_seq_len
        return self.data[start:end:self.sampling_rate]
    
    def export(self, dir:str='.', filename:str='data', split:Tuple | int =1):
        data_to_csv(self, dir=dir, name=filename, split=split)

@deprecated_class
class RICODataset:
    """
    A dataset similar to RicoFullDataset with a split specified by its kind ('train', 'test, 'val) and og size specified by the `ranges` dictionary ('default' -> 0.7, 0.15, 0.15)

    Returns:
    'RICODataset' of specified type
    """
    ranges = {
        'train':0.7,
        'test': 0.15,
        'val': 0.15
              }
    def __init__(self, dataset:RICORawDataset, kind:str, ranges='default', get_every=1) -> None:
        self.kind = kind
        self.get_every = get_every
        if ranges != 'default':
            self.ranges = ranges
        
        if kind == 'train':
            start = 0
            end = int(self.ranges['train']* len(dataset))
        elif kind == 'val':
            start = int(self.ranges['train'] * len(dataset))
            end = int((self.ranges['train'] + self.ranges['val']) * len(dataset))
        elif kind == 'test':
            start = int((self.ranges['train'] + self.ranges['val']) * len(dataset))
            end = len(dataset)
        elif kind == 'full':
            start = 0
            end = len(dataset)
        else:
            raise ValueError(f'kind should be one of "train", "val", "test" or "full" but got {kind}')

        self.data = torch.stack([dataset[i][::self.get_every] for i in range(start, end)])
        self._len = len(self.data)
    def __getitem__(self, idx):
        return self.data[idx]
    def __len__(self):
        return self._len
    def get_kind(self) -> str:
        return self.kind

def data_to_csv(data: RICODataset | RICORawDataset, dir:str, name: str, split=(0.7, 1), header=True) -> None:
    """
    Save a numpy array to two tsv file (0.7 train, 0.15 test) by default

    Parameters
    ----------
    - data (RICODataset or RICORawDataset): The data to save.
    - dir (str): The directory path to save the csv files.
    - name (str): The name of the csv files.
    - split (tuple, optional): Numbers between 0 and 1 defining the splitting coefficient (eg. `(0.7, 1.0)`). 
      If split = 1, then only one full dataset is exported. Default is (0.7, 0.85).
    - header (bool, optional): Whether to include the header in the csv files. Default is True.

    Raises
    ------
    - ValueError: If the input data is not a two-dimensional array.

    Returns
    -------
    - None: This function does not return anything.

    Example usage
    -------------
    data_to_csv(dataset, '/path/to/save', 'my_data', split=(0.7, 1), header=True)
    """
    assert data.data.dim() == 2, ValueError('Only two-dimensional arrays are supported')
    
    array = [row.squeeze().numpy() for row in data]
    df = pd.DataFrame(array)
    df.columns = ['ts_' + str(i) for i in range(len(df.columns))]
    if split == 1:
        if not name.endswith('.csv'):
            name = os.path.splitext(name)[0] + '.csv'

        df.to_csv(os.path.join(dir, name), index=False, header=header)
        return

    train_end = split[0]
    test_end = split[1]

    train_df = df.iloc[:int(train_end*len(df))]
    test_df = df.iloc[int(train_end*len(df)):int(test_end*len(df))]

    # Managing paths
    if not os.path.exists(dir):
        os.makedirs(dir)

    train_path = os.path.join(dir, name.replace('.csv', '') + '_TRAIN.csv')
    test_path = os.path.join(dir, name.replace('.csv', '') + '_TEST.csv')

    # Saving
    train_df.to_csv(train_path, header=None)
    test_df.to_csv(test_path, header=None)