import pandas as pd
import seaborn as sns
import logging
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import pickle

from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import GridSearchCV
from imblearn.over_sampling import SMOTE
from sklearn.metrics import precision_score, recall_score, auc, confusion_matrix, roc_curve, precision_recall_curve

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

        X_train = train.drop(['failed'], axis=1)
        X_test = test.drop(['failed'], axis=1)
        y_train = train.failed
        y_test = test.failed

        # SMOTE 
        X_train_resampled, y_train_resampled = SMOTE(random_state=config['smote_random_state']).fit_resample(X_train, y_train)

        # save column names for plotting feature importance
        col_names = X_train_resampled.columns

        numerical_columns = [col for col in X_train_resampled.columns if col not in df_encoded_columns]

        # Create scaler
        scaler = StandardScaler()
        # Fit the scaler to the numerical columns of the training data
        scaler.fit(X_train_resampled[numerical_columns])
        # Transform the numerical columns of the train, validation, and test set
        X_train_resampled[numerical_columns] = scaler.transform(X_train_resampled[numerical_columns])
        X_test[numerical_columns] = scaler.transform(X_test[numerical_columns])

        # grid search hyperparameter
        XGB_model = XGBClassifier(random_state=config['random_state'])
        search_grid = {'n_estimators':config['XGB']['n_estimators'],
                    'learning_rate':config['XGB']['learning_rate'],
                    'max_depth':config['XGB']['max_depth'],
                    'gamma': config['XGB']['gamma'],
                    'subsample': config['XGB']['subsample'],
                    'colsample_bytree': config['XGB']['colsample_bytree']}

        search = GridSearchCV(estimator=XGB_model,
                            param_grid=search_grid,
                            scoring='recall',
                            n_jobs=config['XGB']['n_jobs'],
                            cv=config['XGB']['cv'],
                            verbose=config['XGB']['verbose'])
        
        logger.info(f'{key} XGB Grid Search Started')
        search.fit(X_train_resampled,y_train_resampled)
        logger.info(f'{key} XGB Grid Search Finished')
        logger.info(f"{key} XGB Best Hyperparameters:{search.best_params_}")

        # fit the final model
        best_xgb = XGBClassifier(random_state=config['random_state'], objective='binary:logistic', **search.best_params_)
        logger.info(f"{key} XGB Model Fitting Started")
        best_xgb.fit(X_train_resampled, y_train_resampled)
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

        # plot 
        fig_cm=plt.figure()
        cf_matrix = confusion_matrix(y_test, y_pred)
        sns.heatmap(cf_matrix, annot=True, cmap='Blues', fmt='g')
        plt.xlabel('Predicted Labels')
        plt.ylabel('True Labels')
        plt.title(f'{key}_XGB_Confusion Matrix')
        save_figure(fig_cm, f"{key}_XGB_cm.png", path)

        # Plot ROC curve
        y_pred_prob = best_xgb.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
        roc_auc = auc(fpr, tpr)

        plt.figure()
        plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'{key}_XGB_ROC_Curve')
        plt.legend(loc="lower right")
        save_figure(plt.gcf(), f"{key}_XGB_roc_curve.png", path)

        # Plot precision-recall curve
        precision, recall, _ = precision_recall_curve(y_test, y_pred_prob)
        pr_auc = auc(recall, precision)

        plt.figure()
        plt.plot(recall, precision, label='Precision-Recall curve (area = %0.2f)' % pr_auc)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'{key}_XGB_Precision-Recall Curve')
        plt.legend(loc="lower left")
        save_figure(plt.gcf(), f"{key}_XGB_precision_recall_curve.png", path)

        # feature importance
        fig = plt.figure()
        sorted_xgbidx = best_xgb.feature_importances_.argsort()
        plt.barh(col_names[sorted_xgbidx], best_xgb.feature_importances_[sorted_xgbidx])
        plt.xlabel("XGBoost Feature Importance")
        plt.title(f'{key} Feature Importance')
        save_figure(fig, f"{key}_XGB_feature_importance.png", path)

        # populate results dictionary
        XGB_result[key]['Recall']=recall_score(y_test, y_pred)
        XGB_result[key]['Precision']=precision_score(y_test, y_pred)
        XGB_result[key]['AUC'] = roc_auc
        XGB_result[key]['TMO'] = best_xgb

        logger.info(f"{key} XGBoost Recall: {recall_score(y_test, y_pred)}, \
                    Precision: {precision_score(y_test, y_pred)}, \
                    AUC:{roc_auc}")
    
    return XGB_result
