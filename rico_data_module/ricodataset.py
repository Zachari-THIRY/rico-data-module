import torch
import pandas as pd

from .base import BaseDataset
from .types import DatasetParams

class RICODataset(BaseDataset):
    def __init__(self, params:DatasetParams) -> None:
        super().__init__()
        self.load(params)
    
    def load(self, params:DatasetParams):
        """
        Load the dataset
        """
        self.name = params["name"]
        columns_identifier = params["columns_identifier"]
        # Reading data

        # usecols = range(1,25) if not 'usecols' in params.keys() else params['usecols']
        
        self.data = pd.read_csv(params["data_path"], usecols = lambda col: col.startswith(columns_identifier),dtype='float32')
        # self.data = self.data.drop(self.data.columns[0], axis=1)
        self.data = torch.tensor(self.data.values).unsqueeze(2).float()

        self.update_size()