from typing import TypedDict

class DatasetParams(TypedDict):
    """
    Attributes:
    -----------
    name : str
    data_path : str
    columns_identifier : str
        The prefix of the columns to be used
    """
    name:str
    data_path:str
    columns_identifier:str