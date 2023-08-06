import os
import streamlit as st
import pandas as pd
import glob
import logging
import joblib
import pickle

logger = logging.getLogger(__name__)

@st.cache_resource
def load_artifacts(time_version: str, model_version: str, path: str, config: dict):
    """
    Load the artifacts including model, scaler, encoder, and column names based on the specified time and model versions.

    Args:
        time_version (str): The selected time granularity version.
        model_version (str): The selected model version.
        path (str): The path where the artifacts are stored.
        config (dict): A dictionary containing configuration parameters.

    Returns:
        tuple: A tuple containing the loaded model, scaler, encoder, and column names.

    """
    list_of_folders = glob.glob(os.path.join(path, '*')) # * means all if need specific format then *.csv
    latest_folder = max(list_of_folders, key=os.path.getctime)

    model_path = os.path.join(latest_folder, model_version)

    # find the tmo
    model_files = glob.glob(os.path.join(model_path, f"{time_version}*"))
    valid_tmo = [f for f in model_files if f.endswith(('.pkl', '.h5'))]
    if len(valid_tmo) == 0:
        logger.error("Model file not found.")
        return None
    print(valid_tmo)
    
    model_file = valid_tmo[0] 
    # Depending on the file extension, load it appropriately
    if model_file.endswith('.h5'):
        model = joblib.load(model_file)
    else:
        with open(model_file, 'rb') as f:
            model = pickle.load(f)
    logger.info('User Selected TMO located')

    # Load in time_version based sacler
    scaler = glob.glob(os.path.join(latest_folder, f"{time_version}_scaler.pkl"))[0]
    with open(scaler, 'rb') as f:
        scaler = joblib.load(f)
        logger.info('Scaler Located')

    # load in encoder
    with open(os.path.join(latest_folder, "encoder.pkl"), 'rb') as f:
        encoder = joblib.load(f)
        logger.info('Encoder Located')
    
    # load in data columns
    columns_to_drop=config['classification_columns_to_drop']
    df = pd.read_csv(os.path.join(latest_folder, f"{time_version}_data.csv"), index_col=0)
    # Get the column names as a list
    column_names = [i for i in df.columns.tolist() if i not in columns_to_drop]
    
    return model, scaler, encoder, column_names