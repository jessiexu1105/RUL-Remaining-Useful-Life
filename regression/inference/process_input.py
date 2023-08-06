import pandas as pd
import logging

logger = logging.getLogger(__name__)

def process_input(user_input, scaler, encoder, config):
    """
    Process user input by applying one-hot encoding and scaling.

    Args:
        user_input (dict): The user input dictionary containing feature names and values.
        scaler: The scaler object for scaling the numeric features.
        encoder: The encoder object for one-hot encoding the categorical features.
        config: The configuration parameters for encoding.

    Returns:
        pd.DataFrame: The processed input dataframe.

    """
    # Create dataframe from user input, assume user_input dict
    if type(user_input)==dict:
        input_df = pd.DataFrame([user_input])
        logger.info('DF based on user input created')
    else:
        input_df=user_input
        logger.info('DF Determined')
    input_df['model'] = input_df['model'].astype(int)

    # If there's only one column to encode, reshape it
    if len(config) == 1:
        input_df[config] = input_df[config].values.reshape(-1, 1)

    # Transform the columns to be encoded
    encoded_features = encoder.transform(input_df[config])
    
    # Reconstruct df_encoded_columns during inference
    df_encoded_columns = [f"{config[i]}_{category}" for i, categories in enumerate(encoder.categories_) for category in categories]
    encoded_df = pd.DataFrame(encoded_features, columns=df_encoded_columns)
    logger.info('Encode Completed')

    # Concatenate the encoded columns back to the input dataframe
    input_df = pd.concat([input_df.drop(config, axis=1), encoded_df], axis=1)

    # Scale the input dataframe
    numeric_features = [col for col in input_df.columns if col not in df_encoded_columns]
    input_df[numeric_features] = scaler.transform(input_df[numeric_features])
    logger.info('Standardization Completed')

    return input_df