import pandas as pd
import numpy as np
import logging
import matplotlib.pyplot as plt
import tensorflow as tf
from pathlib import Path
import sys
import joblib

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import GridSearchCV, train_test_split
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv1D, MaxPooling1D, Flatten

logger = logging.getLogger(__name__)

def save_figure(fig, filename, path:Path)-> None:
    """
    Save a matplotlib figure to a specified path.

    Args:
        figure (matplotlib.figure.Figure): The figure to save.
        filename (str): The filename for the saved figure.
        path (str): The directory path to save the figure to.
    """
    # Create the directory if it doesn't exist
    path.mkdir(parents=True, exist_ok=True)
    try:
        # Set the full file path
        file_path = path / filename
        fig.savefig(file_path)
        logger.info("Figure saved to %s", file_path)
    except FileNotFoundError:
        logger.error("Could not find file %s to save figure.", file_path)
    except IsADirectoryError:
        logger.error("Cannot save figure to a directory. Please provide a valid file path.")

def save_model(model, filename, path: Path) -> None:
    """Saves the trained model to the specified file path.

    Args:
        model: The trainedmodel to save.
        filename: desired name that one wants to give to the file
        path (Path): A Path object representing the file path where the model should be saved.
    """
    # Create the directory if it doesn't exist
    path.mkdir(parents=True, exist_ok=True)
    try:
        # Set the full file path
        file_path = path / filename
        # Save the model
        joblib.dump(model, file_path)
        logger.info("Model saved to %s", path)
    except FileNotFoundError:
        logger.error("Could not find file %s to save model.", file_path)
        sys.exit(1)
    except IsADirectoryError:
        logger.error("Cannot save model to a directory. Please provide a valid file path.")
        sys.exit(1)

def build_CNN_model(X_train, dropout, initial_learning_rate, config):
    """
    Build and compile a Convolutional Neural Network (CNN) model.

    Args:
        X_train (ndarray): The training data with shape (samples, timesteps, features).
        dropout (float): The dropout rate for the dropout layers in the model.
        initial_learning_rate (float): The initial learning rate for the Adam optimizer.
        config (dict): A dictionary containing configuration parameters for the model.

    Returns:
        tf.keras.Model: The compiled CNN model.

    """
    model = Sequential([
        Conv1D(64, kernel_size=config['kernel_size'], activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])),
        MaxPooling1D(pool_size=config['pool_size']),
        Dropout(dropout),
        Conv1D(32, kernel_size=config['kernel_size'], activation='relu'),
        MaxPooling1D(pool_size=config['pool_size']),
        Dropout(dropout),
        Flatten(),
        Dense(32, activation='relu'),
        Dense(64, activation='softplus'),
        Dense(1)
    ])
    # Define the learning rate schedule using ExponentialDecay
    model.compile(loss = "mse", optimizer = tf.keras.optimizers.Adam(learning_rate=initial_learning_rate), \
                  metrics=[tf.keras.metrics.MeanSquaredError()])

    return model

def CNN_fitting(time_dimension_dict: dict, config: dict, path: Path)->dict:
    """
    Fit a Convolutional Neural Network (CNN) model on all df in the time_dimension_dict.

    Args:
        time_dimension_dict (dict): The dictionary following the format time_dimension: df.
        config (dict): A dictionary containing configuration parameters for the fitting process.
        path (Path): where to save all artifacts to 

    Returns:
        dict: model evaluation results.
    """
    # define result dictionary
    CNN_result={}

    for key, df in time_dimension_dict.items():
        CNN_result[key]={}

        # Create a DataFrame with only the columns to be one-hot encoded
        columns_to_encode = config['one_hot_encode']
        df_encoded = df[columns_to_encode]

        # Perform one-hot encoding using OneHotEncoder
        encoder = OneHotEncoder(sparse_output=False)
        encoded_features = encoder.fit_transform(df_encoded)

        # Replace the original columns with the encoded features
        df_encoded_columns = [f"{col}_{category}" for col, categories in \
                                zip(columns_to_encode, encoder.categories_) for category in categories]
        df_encoded = pd.DataFrame(encoded_features, columns=df_encoded_columns)

        # Concatenate the encoded features back to the original DataFrame
        df = pd.concat([df.drop(columns_to_encode, axis=1), df_encoded], axis=1)

        # train test split
        train=df[df['Dataset']=='Train'].drop('Dataset', axis=1)
        test=df[df['Dataset']=='Test'].drop('Dataset', axis=1)

        X_train = train.drop(['RUL_piecewise'], axis=1)
        X_test = test.drop(['RUL_piecewise'], axis=1)
        y_train = train.RUL_piecewise
        y_test = test.RUL_piecewise

        # Split the training data further into train and validation
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, \
                                                          test_size=config['validation_size'], \
                                                          random_state=config['random_state'])

        numerical_columns = [col for col in X_train.columns if col not in df_encoded_columns]

        # Create scaler
        scaler = StandardScaler()
        # Fit the scaler to the numerical columns of the training data
        scaler.fit(X_train[numerical_columns])
        # Transform the numerical columns of the train, validation, and test set
        X_train[numerical_columns] = scaler.transform(X_train[numerical_columns])
        X_val[numerical_columns] = scaler.transform(X_val[numerical_columns])
        X_test[numerical_columns] = scaler.transform(X_test[numerical_columns])

        # reshape for CNN model
        num_features=1
        X_train = X_train.values.reshape(X_train.shape[0], X_train.shape[1], num_features)
        X_val = X_val.values.reshape(X_val.shape[0], X_val.shape[1], num_features)
        X_test = X_test.values.reshape(X_test.shape[0], X_test.shape[1], num_features)

        regressor = KerasRegressor(build_fn=build_CNN_model, X_train=X_train, config=config['CNN'], \
                                   epochs=config['CNN']['tuning_epochs'], batch_size=config['CNN']['tuning_batch_size'], \
                                    verbose=config['CNN']['tuning_verbose'])

        # Define the hyperparameters to tune and their respective values
        param_grid = {
            'dropout': config['CNN']['drop_out'],
            'initial_learning_rate': config['CNN']['initial_learning_rate']
        }

        # Perform grid search with cross-validation
        grid = GridSearchCV(estimator=regressor, param_grid=param_grid, cv=config['CNN']['cv'])
        logger.info(f'{key} CNN Grid Search Started')
        grid.fit(X_train, y_train)
        logger.info(f'{key} CNN Grid Search Finished')

        best_params = grid.best_params_
        logger.info(f'{key} CNN Best Hyperparameters: {best_params}')

        # Create the final model with the best hyperparameters
        best_cnn = build_CNN_model(X_train, dropout=best_params['dropout'], \
                                     initial_learning_rate =best_params['initial_learning_rate'], config=config['CNN'])
        
        early_stopping = EarlyStopping(monitor='val_loss', patience=config['CNN']['patience'])

        # fit tuned model on training data
        logger.info(f'{key} CNN Model Fitting Started')
        history = best_cnn.fit(X_train, y_train, validation_data=(X_val, y_val), callbacks=[early_stopping], \
                               epochs=config['CNN']['fitting_epochs'], batch_size=config['CNN']['fitting_batch_size'], \
                                verbose=config['CNN']['fitting_verbose'])
        save_model(best_cnn, f'{key}_cnn.h5', path)

        test_results = best_cnn.evaluate(X_test, y_test)
        test_loss = test_results[0]
        test_rmse = np.sqrt(test_results[1])
        logger.info(f"{key} CNN Test Loss: {test_loss}, RMSE: {test_rmse}")

        # populate results dictionary
        CNN_result[key]['Loss']=test_loss
        CNN_result[key]['RMSE']=test_rmse
        CNN_result[key]['TMO']=best_cnn

        # plot loss evolution
        fig = plt.figure()
        plt.plot(history.history['loss'], label='Loss')
        plt.plot(history.history['val_loss'], label='Val_Loss')
        plt.legend()
        plt.title('Loss Evolution')
        save_figure(fig, f"{key}_CNN_loss_evolution.png", path)

    return CNN_result
