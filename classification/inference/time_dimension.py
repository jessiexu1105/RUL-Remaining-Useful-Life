import pandas as pd
import logging

logger = logging.getLogger(__name__)

def create_time_dimension(df:pd.DataFrame)->dict:
    """
    Create time dimension DataFrames from the input DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        dict: A dictionary containing time dimension DataFrames for different time granularities.
    """
    # always keep these
    columns_to_keep=['machineID','datetime','model','age','RUL']
    # get 24h df
    columns_24h=df.columns[df.columns.str.endswith('24h')].tolist()
    df_24h=df[columns_to_keep+columns_24h]
    logger.info('24h DataFrame Created')

    # get 5d df
    columns_5d=df.columns[df.columns.str.endswith('5d')].tolist()
    df_5d=df[columns_to_keep+columns_5d]
    logger.info('5d DataFrame Created')

    time_dimension_dict={'24h': df_24h, '5d':df_5d}

    return time_dimension_dict


