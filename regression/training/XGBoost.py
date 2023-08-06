import pandas as pd
import numpy as np
import logging
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import pickle

from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error as mse

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

def XGB_fitting(time_dimension_dict: dict, config: dict, path: Path)->dict:
    """
    Fit a XGBoost model on all df in the time_dimension_dict.

    Args:
        time_dimension_dict (dict): The dictionary following the format time_dimension: df.
        config (dict): A dictionary containing configuration parameters for the fitting process.
        path (Path): where to save all artifacts to 

    Returns:
        dict: model evaluation results.
    """
    # define result dictionary
    XGB_result={}

    for key, df in time_dimension_dict.items():
        XGB_result[key]={}

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

        # save column names for plotting feature importance
        col_names = X_train.columns

        numerical_columns = [col for col in X_train.columns if col not in df_encoded_columns]

        # Create scaler
        scaler = StandardScaler()
        # Fit the scaler to the numerical columns of the training data
        scaler.fit(X_train[numerical_columns])
        # Transform the numerical columns of the train, validation, and test set
        X_train[numerical_columns] = scaler.transform(X_train[numerical_columns])
        X_test[numerical_columns] = scaler.transform(X_test[numerical_columns])

        # grid search hyperparameter
        XGB_model = XGBRegressor(random_state=config['random_state'])
        search_grid = {'n_estimators':config['XGB']['n_estimators'],
                    'learning_rate':config['XGB']['learning_rate'],
                    'max_depth':config['XGB']['max_depth'],
                    'gamma': config['XGB']['gamma'],
                    'subsample': config['XGB']['subsample']}
        
        search = GridSearchCV(estimator=XGB_model,
                            param_grid=search_grid,
                            scoring='neg_mean_squared_error',
                            n_jobs=config['XGB']['n_jobs'],
                            cv=config['XGB']['cv'],
                            verbose=config['XGB']['verbose'])
        logger.info(f'{key} XGB Grid Search Started')
        search.fit(X_train,y_train)
        logger.info(f'{key} XGB Grid Search Finished')
        logger.info(f"{key} XGB Best Hyperparameters:{search.best_params_}")

        # fit the final model
        best_xgb = XGBRegressor(random_state=config['random_state'], **search.best_params_)
        logger.info(f"{key} XGB Model Fitting Started")
        best_xgb.fit(X_train, y_train)
        logger.info(f"{key} XGB Model Fitting Finished")

        # Save the model
        path.mkdir(parents=True, exist_ok=True)
        try:
            with open(path / f'{key}_xgb.pkl', 'wb') as file:
                pickle.dump(best_xgb, file)
            logger.info(f"XGB Model saved to {path}")
        except Exception as e:
            logger.error(f"Failed to save XGB model. Reason: {e}")
            sys.exit(1)

        # predict the test result
        y_pred = best_xgb.predict(X_test)

        test_rmse = np.sqrt(mse(y_test, y_pred))
        logger.info(f"{key} XGB RMSE: {test_rmse}")

        # populate results dictionary
        XGB_result[key]['RMSE']=test_rmse
        XGB_result[key]['TMO']=best_xgb

        # feature importance
        fig = plt.figure()
        sorted_xgbidx = best_xgb.feature_importances_.argsort()
        plt.barh(col_names[sorted_xgbidx], best_xgb.feature_importances_[sorted_xgbidx])
        plt.xlabel("XGBoost Feature Importance")
        plt.title(f'{key} Feature Importance')
        save_figure(fig, f"{key}_XGB_feature_importance.png", path)
    
    return XGB_result
