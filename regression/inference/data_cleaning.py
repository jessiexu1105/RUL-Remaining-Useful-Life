import logging
import pandas as pd

logger = logging.getLogger(__name__)

def data_cleaning(df)->pd.DataFrame:
    """
    Perform data cleaning operations on the DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame to be cleaned.

    Returns:
        pd.DataFrame: The cleaned DataFrame.
    """
    # dtype conversion
    df['datetime']=pd.to_datetime(df['datetime'], format='%Y-%m-%d')
    df['model']=df['model'].replace(['model1','model2','model3','model4'], [1,2,3,4])
    df['failure']=df['failure'].replace(['none','comp1','comp2','comp3','comp4'], [0,1,2,3,4])
    
    # check and drop na values
    columns_with_null=[]
    for column in df.columns:
        if df[column].isna().any():
            columns_with_null.append(column)
    if len(columns_with_null)==0:
        logger.info('All Columns Have Valid Inputs')
    else:
        df=df.dropna()
        logger.info(f'{columns_with_null} with null entries detected, invalid entries droppped')

    # check and drop duplicated columns
    duplicated_columns = df.loc[:, df.T.duplicated()].columns.tolist()
    if len(duplicated_columns)==0:
        logger.info('No Duplicated Columns Detected')
    else:
        df=df.drop(columns=duplicated_columns, axis=1)
        logger.info(f'Duplicated column(s) {duplicated_columns} detected, dropped')
    
    return df

def drop_negative(df, config:dict)->pd.DataFrame:
    """
    Check for negative values in specified numeric columns and drop corresponding rows.

    Args:
        df (pd.DataFrame): The input DataFrame.
        config (dict): A dictionary containing configuration parameters.

    Returns:
        pd.DataFrame: The DataFrame with rows containing negative values dropped.
    """
    # check and drop negative values
    columns_with_negative=[]
    for i in range(config['numeric_column_start'], config['numeric_column_end']+1):
        if (df.iloc[:, i] < 0).any():
                columns_with_negative.append(df.columns[i])  
    if len(columns_with_negative)==0:
        logger.info('No Negative Values Detected')
    else:
        df = df.loc[~(df[columns_with_negative] < 0).any(axis=1)]
        logger.info(f'{columns_with_negative} with negative values detected, invalid entries droppped')
    
    return df
        
