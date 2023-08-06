import pandas as pd
from pathlib import Path
import sys
import joblib
import logging

logger = logging.getLogger(__name__)

def save_model(model, filename, path: Path) -> None:
    """Saves the trained random forest model to the specified file path.

    Args:
        model (sklearn.ensemble.RandomForestClassifier): The trained random forest model to save.
        save_path (Path): A Path object representing the file path where the model should be saved.
    """
    # Create the directory if it doesn't exist
    path.mkdir(parents=True, exist_ok=True)
    try:
        # Set the full file path
        file_path = path / filename
        # Save the model
        joblib.dump(model, file_path)
        logger.info("Best Model saved to %s", path)
    except FileNotFoundError:
        logger.error("Could not find file %s to save model.", file_path)
        sys.exit(1)
    except IsADirectoryError:
        logger.error("Cannot save model to a directory. Please provide a valid file path.")
        sys.exit(1)

def model_comparison(model_dict, path: Path, config:dict)->pd.DataFrame:
    """
    Compare metrics across different models and time versions and save the comparison DataFrame to a file.

    Args:
        model_dict (dict): A dictionary containing model results for different model names.
        path (Path): The path where the comparison DataFrame will be saved.
        config (dict): A dictionary containing configuration parameters.

    Returns:
        pd.DataFrame: The comparison DataFrame.
    """
    comparison_df=pd.DataFrame(columns=['Model', 'Metric']+list(list(model_dict.values())[0].keys()))
    metric_to_compare=config['metric_to_compare']
    comparison_df['Model'] = list(item for item in list(model_dict.keys()) for _ in range(len(metric_to_compare)))
    comparison_df['Metric'] = list(metric_to_compare)*len(model_dict)

    for time in list(list(model_dict.values())[0].keys()):
        result_list=[]
        for model in list(model_dict.values()):
            for key, metrics in model.items():
                for metric_name, value in metrics.items():
                    if key==time and metric_name in metric_to_compare:
                        result_list.append(value)
        comparison_df[time]=result_list
    
    comparison_df.set_index(['Model', 'Metric'], inplace=True)
    save_path = path / 'comparison_df.csv'
    comparison_df.to_csv(save_path)

    logger.info(f'Comparison DF Created and Saved to {save_path}')

    return comparison_df

def locate_best_model(df: pd.DataFrame, model_dict: dict, data_dict: dict, config: dict, path:Path)->None:
    """
    Locate the best model based on the provided DataFrame and model dictionary,
    and save the best model and corresponding data for inference.

    Args:
        df (pd.DataFrame): The input DataFrame containing model comparison results.
        model_dict (dict): A dictionary containing model result dicts for different models.
        data_dict (dict): A dictionary containing training data for different time versions.
        config (dict): A dictionary containing configuration parameters.
        path (Path): The path where the best model and data will be saved.

    Returns:
        None
    """
    main_metric = config['main_metric']
    df_filtered = df.xs(main_metric, level='Metric')
    for column in df_filtered.columns:
        row_with_max = df_filtered[column].idxmax()
        best_model=model_dict[f'{row_with_max}'][f'{column}']['TMO']
        # save best tmo
        if row_with_max in ['LSTM']:
            save_model(best_model, f'{column}_best_model.h5', path / 'Best_Model')
        else:
            save_model(best_model, f'{column}_best_model.pkl', path / 'Best_Model') 

        # also save data for read in columns in inference
        training_data = data_dict[column]
        training_data.to_csv(path / f'{column}_data.csv')
        logger.info(f"{column} data have been copied to {path}") 
