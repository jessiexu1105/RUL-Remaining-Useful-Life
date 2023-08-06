import pandas as pd
import numpy as np
import seaborn as sns
import logging
import matplotlib.pyplot as plt
import tensorflow as tf
from pathlib import Path
import sys
import joblib

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve, auc
from imblearn.over_sampling import SMOTE
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout

logger = logging.getLogger(__name__)

def save_preprocessing_artifacts(key, encoder, scaler, path: Path)-> None:
    """Saves preprocessing artifacts to the specified file path.

    Args:
        key: which time_granularity is it for
        encoder: The encoder object.
        scaler: The scaler object.
        path (Path): A Path object representing the directory where the artifacts should be saved.
    """
    path.mkdir(parents=True, exist_ok=True)
    try:
        joblib.dump(encoder, path / 'encoder.pkl')
        joblib.dump(scaler, path / f'{key}_scaler.pkl')
        logger.info(f"{key} Encoder and Scaler saved to %s", path)
    except Exception as e:
        logger.error(f"Failed to save preprocessing artifacts. Reason: {e}")
        sys.exit(1)

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
    """Saves the trained random forest model to the specified file path.

    Args:
        model (sklearn.ensemble.RandomForestClassifier): The trained random forest model to save.
        filename (str): desired filename for the model object
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

def build_LSTM_model(X_train, dropout, initial_learning_rate):
    """
    Build and compile a Long Short-Term Memory (LSTM) model.

    Args:
        X_train (ndarray): The training data with shape (samples, timesteps, features).
        dropout (float): The dropout rate for the dropout layers in the model.
        initial_learning_rate (float): The initial learning rate for the Adam optimizer.
        config (dict): A dictionary containing configuration parameters for the model.

    Returns:
        tf.keras.Model: The compiled LSTM model.
    """
    model = Sequential([
        LSTM(64, input_shape = (X_train.shape[1], X_train.shape[2]), activation = "relu", return_sequences=True),
        Dropout(dropout),
        LSTM(32, activation = "relu", return_sequences = True),
        Dropout(dropout),
        LSTM(16, activation = "relu", return_sequences=False),
        Dense(32, activation = "relu"),
        Dense(64, activation = "relu"),
        Dense(1, activation = "sigmoid")
    ])
    # attempt 1: Define the learning rate schedule using ExponentialDecay
    # not working: both training and validation loss stay constant
    # lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    #     initial_learning_rate,
    #     decay_steps=config['decay_steps'],
    #     decay_rate=config['decay_rate'],
    #     staircase=True
    # )

    model.compile(loss=tf.keras.losses.BinaryCrossentropy(), \
                  optimizer = tf.keras.optimizers.Adam(learning_rate=initial_learning_rate), \
                  metrics=[tf.keras.metrics.Recall(),tf.keras.metrics.Precision(), tf.keras.metrics.AUC()])

    return model

def lstm_fitting(time_dimension_dict: dict, config: dict, path: Path)->dict:
    """
    Fit a Long Short-Term Memory (LSTM) model on all df in the time_dimension_dict.

    Args:
        time_dimension_dict (dict): The dictionary following the format time_dimension: df.
        config (dict): A dictionary containing configuration parameters for the fitting process.
        path (Path): where to save all artifacts to 

    Returns:
        dict: model evaluation results.
    """
    # define result dictionary
    LSTM_result={}

    for key, df in time_dimension_dict.items():
        LSTM_result[key]={}

        # Create a DataFrame with only the columns to be one-hot encoded
        columns_to_encode = config['one_hot_encode']
        df_encoded = df[columns_to_encode]

        if len(columns_to_encode) == 1:
            df_encoded = df_encoded.values.reshape(-1, 1)

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

        X_train = train.drop(['failed'], axis=1)
        X_test = test.drop(['failed'], axis=1)
        y_train = train.failed
        y_test = test.failed

        # Split the training data further into train and validation
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, \
                                                          test_size=config['validation_size'], \
                                                          random_state=config['random_state'])

        numerical_columns = [col for col in X_train.columns if col not in df_encoded_columns]

        # SMOTE 
        X_train_resampled, y_train_resampled = SMOTE(random_state=config['smote_random_state']).fit_resample(X_train, y_train)

        # Create scaler
        scaler = StandardScaler()
        # Fit the scaler to the numerical columns of the training data
        scaler.fit(X_train[numerical_columns])
        # Transform the numerical columns of the train, validation, and test set
        X_train_resampled[numerical_columns] = scaler.transform(X_train_resampled[numerical_columns])
        X_val[numerical_columns] = scaler.transform(X_val[numerical_columns])
        X_test[numerical_columns] = scaler.transform(X_test[numerical_columns])

        # save all preprocessing artifacts 
        save_preprocessing_artifacts(key, encoder, scaler, path)

        # reshape for LSTM model
        num_features = 1
        X_train_resampled = X_train_resampled.values.reshape(X_train_resampled.shape[0], X_train_resampled.shape[1], num_features)
        X_val = X_val.values.reshape(X_val.shape[0], X_val.shape[1], num_features)
        X_test = X_test.values.reshape(X_test.shape[0], X_test.shape[1], num_features)

        # define wrapper for grid search
        regressor = KerasRegressor(build_fn=build_LSTM_model, X_train=X_train_resampled, \
                                   epochs=config['LSTM']['tuning_epochs'], batch_size=config['LSTM']['tuning_batch_size'], \
                                    verbose=config['LSTM']['tuning_verbose'])

        # Define the hyperparameters to tune and their respective values
        param_grid = {
            'dropout': config['LSTM']['drop_out'],
            'initial_learning_rate': config['LSTM']['initial_learning_rate']
        }

        # Perform grid search with cross-validation
        grid = GridSearchCV(estimator=regressor, param_grid=param_grid, cv=config['LSTM']['cv'])
        logger.info(f'{key} LSTM Grid Search Started')
        grid.fit(X_train_resampled, y_train_resampled)
        logger.info(f'{key} LSTM Grid Search Finished')

        best_params = grid.best_params_
        logger.info(f"{key} LSTM Best Hyperparameters: {best_params}")

        # Create the final model with the best hyperparameters
        best_lstm = build_LSTM_model(X_train_resampled, dropout=best_params['dropout'], \
                                     initial_learning_rate =best_params['initial_learning_rate'])
        
        # attempt 2: lr scheduled based on epoch
        # def scheduler(epoch, lr):
        #     if epoch < 25:
        #         return lr
        #     else:
        #         return lr*0.5 (also tried 0.8)
        # lr_callback = LearningRateScheduler(scheduler)
        # both training and validation loss stuck after 25 epoch -> learning rate too small
        
        # attempt 3: define early stopping
        early_stopping = EarlyStopping(monitor='val_loss', patience=config['LSTM']['patience']) # also tried patience 10

        # Fit the best model on the combined training and validation data
        logger.info(f"{key} LSTM Model Fitting Started")
        history = best_lstm.fit(X_train_resampled, y_train_resampled, validation_data=(X_val, y_val), callbacks=[early_stopping], 
                                epochs=config['LSTM']['fitting_epochs'], batch_size=config['LSTM']['fitting_batch_size'], \
                                verbose=config['LSTM']['fitting_verbose'])
        save_model(best_lstm, f'{key}_lstm.h5', path / 'LSTM')

        test_results = best_lstm.evaluate(X_test, y_test)
        test_loss = test_results[0]
        test_recall = np.sqrt(test_results[1])
        test_precision = test_results[2]
        test_auc = test_results[3]
        logger.info(f"{key} LSTM Test Loss: {test_loss}, Recall: {test_recall}, Precision: {test_precision}, AUC:{test_auc}")

        # populate results dictionary
        LSTM_result[key]['Loss']=test_loss
        LSTM_result[key]['Recall']=test_recall
        LSTM_result[key]['Precision']=test_precision
        LSTM_result[key]['AUC']=test_auc
        LSTM_result[key]['TMO']=best_lstm

        # plot loss evolution
        fig = plt.figure()
        plt.plot(history.history['loss'], label='Loss')
        plt.plot(history.history['val_loss'], label='Val_Loss')
        plt.legend()
        plt.title(f'{key} Loss Evolution')
        save_figure(fig, f"{key}_LSTM_loss_evolution.png", path / 'LSTM')

        # plot confusion matrix
        y_pred = np.round(best_lstm.predict(X_test)).flatten()

        # Compute the confusion matrix
        cm = confusion_matrix(y_test, y_pred)

        # Plot the confusion matrix as a heatmap
        fig_cm=plt.figure()
        sns.heatmap(cm, annot=True, cmap='Blues', fmt='g')
        plt.xlabel('Predicted Labels')
        plt.ylabel('True Labels')
        plt.title(f'{key}_Confusion Matrix')
        save_figure(fig_cm, f"{key}_LSTM_cm.png", path / 'LSTM')

        # Plot ROC curve
        y_pred_prob = best_lstm.predict(X_test)
        fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
        roc_auc = auc(fpr, tpr)

        plt.figure()
        plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'{key}_LSTM_ROC_Curve')
        plt.legend(loc="lower right")
        save_figure(plt.gcf(), f"{key}_LSTM_roc_curve.png", path / 'LSTM')

        # Plot precision-recall curve
        precision, recall, _ = precision_recall_curve(y_test, y_pred_prob)
        pr_auc = auc(recall, precision)

        plt.figure()
        plt.plot(recall, precision, label='Precision-Recall curve (area = %0.2f)' % pr_auc)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'{key}_LSTM_Precision-Recall Curve')
        plt.legend(loc="lower left")
        save_figure(plt.gcf(), f"{key}_LSTM_precision_recall_curve.png", path / 'LSTM')

    return LSTM_result
