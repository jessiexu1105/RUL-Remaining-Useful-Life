import pandas as pd
import logging
import matplotlib.pyplot as plt
from pathlib import Path
import pickle
import sys
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import GridSearchCV
from imblearn.over_sampling import SMOTE
from sklearn.metrics import precision_score, recall_score, auc, confusion_matrix, roc_curve, precision_recall_curve, roc_auc_score

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

def Logistic_fitting(time_dimension_dict: dict, config: dict, path: Path)->dict:
    """
    Fit a Logistic Regression model on all df in the time_dimension_dict.

    Args:
        time_dimension_dict (dict): The dictionary following the format time_dimension: df.
        config (dict): A dictionary containing configuration parameters for the fitting process.
        path (Path): where to save all artifacts to 

    Returns:
        dict: model evaluation results.
    """
    # define result dictionary
    Logistic_result={}

    for key, df in time_dimension_dict.items():
        Logistic_result[key]={}

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

        numerical_columns = [col for col in X_train.columns if col not in df_encoded_columns]

        # Create scaler
        scaler = StandardScaler()
        # Fit the scaler to the numerical columns of the training data
        scaler.fit(X_train[numerical_columns])
        # Transform the numerical columns of the train, validation, and test set
        X_train[numerical_columns] = scaler.transform(X_train[numerical_columns])
        X_test[numerical_columns] = scaler.transform(X_test[numerical_columns])

        Logistic = LogisticRegression(random_state=config['random_state'], max_iter=5000)
        # Logistic.fit(X_train, y_train)

        param_grid = {'C': config['Logistic']['C'], 'penalty': config['Logistic']['penalty']}
        grid = GridSearchCV(estimator=Logistic, param_grid=param_grid, cv= config['Logistic']['cv'])
        logger.info(f'{key} Logistic Grid Search Started')
        grid.fit(X_train_resampled, y_train_resampled)
        logger.info(f'{key} Logistic Grid Search Finished')
        best_params = grid.best_params_
        logger.info(f'{key} Logistic Best Hyperparameters: {best_params}')
        lr_best = LogisticRegression(**best_params, random_state=config['random_state'], max_iter=5000) # based on best_params_
        lr_best.fit(X_train_resampled, y_train_resampled)

        # Save the model
        path.mkdir(parents=True, exist_ok=True)
        try:
            with open(path / f'{key}_logistic.pkl', 'wb') as file:
                pickle.dump(lr_best, file)
            logger.info(f"Logistic Model saved to {path}")
        except Exception as e:
            logger.error(f"Failed to save Logistic model. Reason: {e}")
            sys.exit(1)

        # plot confusion matrix
        y_pred = lr_best.predict(X_test)

        fig_cm=plt.figure()
        cf_matrix = confusion_matrix(y_test, y_pred)
        sns.heatmap(cf_matrix, annot=True, cmap='Blues', fmt='g')
        plt.xlabel('Predicted Labels')
        plt.ylabel('True Labels')
        plt.title(f'{key}_Logistic_Confusion Matrix')
        save_figure(fig_cm, f"{key}_Logistic_cm.png", path)

        # Plot ROC curve
        y_pred_prob = lr_best.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
        roc_auc = auc(fpr, tpr)

        plt.figure()
        plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'{key}_Logistic_ROC_Curve')
        plt.legend(loc="lower right")
        save_figure(plt.gcf(), f"{key}_Logistic_roc_curve.png", path)

        # Plot precision-recall curve
        precision, recall, _ = precision_recall_curve(y_test, y_pred_prob)
        pr_auc = auc(recall, precision)

        plt.figure()
        plt.plot(recall, precision, label='Precision-Recall curve (area = %0.2f)' % pr_auc)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'{key}_Logistic_Precision-Recall Curve')
        plt.legend(loc="lower left")
        save_figure(plt.gcf(), f"{key}_Logistic_precision_recall_curve.png", path)

        # feature importance
        feature_importances = pd.DataFrame({'feature': X_train_resampled.columns, 'importance': lr_best.coef_[0]})
        feature_importances = feature_importances.sort_values('importance', ascending=False)

        plt.figure()
        plt.barh(feature_importances['feature'], feature_importances['importance'])
        plt.xlabel('Feature Importance')
        plt.title(f'{key} Logistic Model')
        save_figure(plt.gcf(), f"{key}_Logistic_feature_importance.png", path)

        # populate results dictionary
        Logistic_result[key]['Recall']=recall_score(y_test, y_pred)
        Logistic_result[key]['Precision']=precision_score(y_test, y_pred)
        Logistic_result[key]['AUC'] = roc_auc
        Logistic_result[key]['TMO'] = lr_best

        logger.info(f"{key} Logistic Recall: {recall_score(y_test, y_pred)}, \
                    Precision: {precision_score(y_test, y_pred)}, \
                    AUC:{roc_auc}")

    return Logistic_result
