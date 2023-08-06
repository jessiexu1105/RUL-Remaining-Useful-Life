import pandas as pd

def load_data(train_path:str, test_path:str, config:dict)->pd.DataFrame:
    """
    Load and merge the train and test data from the specified paths.

    Args:
        train_path (str): The file path of the train data.
        test_path (str): The file path of the test data.
        config (dict): A dictionary containing configuration parameters for data loading.

    Returns:
        pd.DataFrame: The merged DataFrame containing train and test data.
    """
    train=pd.read_csv(train_path)
    test=pd.read_csv(test_path)

    merge=pd.concat([train, test], axis=0, ignore_index=True)
    merge['Dataset']=['Train']*len(train)+['Test']*len(test)
    merge=merge.sort_values(by=config['order_by'], ignore_index=True)

    return merge