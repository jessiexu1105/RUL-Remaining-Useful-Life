import pandas as pd

def load_data(path:str, config:dict)->pd.DataFrame:
    """
    Load data from the specified path.

    Args:
        path (str): The file path of the data.
        config (dict): A dictionary containing configuration parameters for data loading.

    Returns:
        pd.DataFrame: The sorted data.
    """
    df=pd.read_csv(path)
    df=df.sort_values(by=config['order_by'], ignore_index=True)

    return df