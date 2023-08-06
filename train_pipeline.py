import argparse
import datetime
import logging.config
from pathlib import Path

import yaml

from classification.classification_training import load_data as ld, \
    data_cleaning as dc, time_dimension as td_c, \
    LSTM as LSTM_c, Logistic as Logistic_c, XGBoost as XGBoost_c, \
    model_comparison as mc_c
from regression.regression_training import feature_engineering as fe, \
    time_dimension as td_r, \
    LSTM as LSTM_r, CNN as CNN_r, XGBoost as XGBoost_r, \
    model_comparison as mc_r


logging.config.fileConfig("config/local.conf")
logger_classification_training = logging.getLogger("classification_training")
logger_classification = logging.getLogger("classification")
logger_regression_training = logging.getLogger("regression_training")
logger_regression = logging.getLogger("regression")

if __name__ == "__main__":
    parser_classification = argparse.ArgumentParser(
        description="Acquire, clean, and create features from RUL data"
    )
    parser_classification.add_argument(
        "--config", default="config/classification-config.yaml", help="Path to configuration file"
    )
    args_classification = parser_classification.parse_args()

    # Load configuration file for parameters and run config
    with open(args_classification.config, "r") as f:
        try:
            classification_config = yaml.load(f, Loader=yaml.FullLoader)
        except yaml.error.YAMLError as e:
            logger_classification_training.error("Error while loading configuration from %s", args_classification.config)
        else:
            logger_classification_training.info("Configuration file loaded from %s", args_classification.config)

    run_config = classification_config.get("run_config", {})

    # Set up output directory for saving classification artifacts
    now = int(datetime.datetime.now().timestamp())
    classification_artifacts = Path(run_config.get("output", "classification_runs")) / str(now)
    classification_artifacts.mkdir(parents=True)

    # Add a FileHandler for the artifact directory to your logger
    classification_file_handler = logging.FileHandler(classification_artifacts / "classification.log")
    classification_file_handler.setLevel(logging.DEBUG)  # Or any level you want
    classification_file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger_classification.addHandler(classification_file_handler)

    # Save config file to artifacts directory for traceability
    with (classification_artifacts / "classification_config.yaml").open("w") as f:
        yaml.dump(classification_config, f)
    
    # load in datasets
    # df=ld.load_data('data/ALLtrainMescla5D.csv', 'data/ALLtestMescla5D.csv', classification_config['load_data'])
    df=ld.load_data('data/sample_train.csv', 'data/sample_test.csv', classification_config['load_data'])

    # clean data
    data_cleaning_config = classification_config.get("data_cleaning")
    if data_cleaning_config.get("detect_negative", False):
        df=dc.drop_negative(df, classification_config['data_cleaning'])
    df_cleaned=dc.data_cleaning(df)

    # generate new RUL feature
    # df_cleaned['RUL_piecewise']=fe.Piecewise_RUL(df_cleaned, config['feature_engineering'])

    # create different time dimension df
    classification_time_dimension_dict=td_c.create_time_dimension(df_cleaned)

    # enter model training
    lstm_classification_result = LSTM_c.lstm_fitting(classification_time_dimension_dict, classification_config['model_training'], classification_artifacts)
    logistic_classification_result = Logistic_c.Logistic_fitting(classification_time_dimension_dict, classification_config['model_training'], classification_artifacts / 'Logistic')
    xgb_classification_result = XGBoost_c.XGB_fitting(classification_time_dimension_dict, classification_config['model_training'], classification_artifacts / 'XGB') 

    classification_model_dict={'LSTM': lstm_classification_result, 'Logistic': logistic_classification_result, 'XGBoost': xgb_classification_result}

    # model comparison
    classification_comparison_df=mc_c.model_comparison(classification_model_dict, classification_artifacts, classification_config['model_comparison'])
    mc_c.locate_best_model(classification_comparison_df, classification_model_dict, classification_time_dimension_dict, classification_config['model_comparison'], classification_artifacts)

    # start regression 
    parser_regression = argparse.ArgumentParser(
        description="Acquire, clean, and create features from RUL data"
    )
    parser_regression.add_argument(
        "--config", default="config/regression-config.yaml", help="Path to configuration file"
    )
    args_regression = parser_regression.parse_args()

    # Load configuration file for parameters and run config
    with open(args_regression.config, "r") as f:
        try:
            regression_config = yaml.load(f, Loader=yaml.FullLoader)
        except yaml.error.YAMLError as e:
            logger_regression_training.error("Error while loading configuration from %s", args_regression.config)
        else:
            logger_regression_training.info("Configuration file loaded from %s", args_regression.config)

    run_config = regression_config.get("run_config", {})

    # Set up output directory for saving artifacts
    regression_artifacts = Path(run_config.get("output", "regression_runs")) / str(now)
    regression_artifacts.mkdir(parents=True)

    # Add a FileHandler for the artifact directory to your logger
    regression_file_handler = logging.FileHandler(regression_artifacts / "regression.log")
    regression_file_handler.setLevel(logging.DEBUG)  # Or any level you want
    regression_file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger_regression.addHandler(regression_file_handler)

    # Save config file to artifacts directory for traceability
    with (regression_artifacts / "regression_config.yaml").open("w") as f:
        yaml.dump(regression_config, f)
    
    # # load in datasets
    # df=ld.load_data('data/ALLtrainMescla5D.csv', 'data/ALLtestMescla5D.csv', regression_config['load_data'])

    # # clean data
    # data_cleaning_config = regression_config.get("data_cleaning")
    # if data_cleaning_config.get("detect_negative", False):
    #     df=dc.drop_negative(df, regression_config['data_cleaning'])
    # df_cleaned=dc.data_cleaning(df)

    # generate new RUL feature
    df_cleaned['RUL_piecewise']=fe.Piecewise_RUL(df_cleaned, regression_config['feature_engineering'])

    # create different time dimension df
    regression_time_dimension_dict=td_r.create_time_dimension(df_cleaned)

    # enter model training
    lstm_regression_result = LSTM_r.lstm_fitting(regression_time_dimension_dict, regression_config['model_training'], regression_artifacts)
    cnn_regression_result = CNN_r.CNN_fitting(regression_time_dimension_dict, regression_config['model_training'], regression_artifacts / 'CNN')
    xgb_regression_result = XGBoost_r.XGB_fitting(regression_time_dimension_dict, regression_config['model_training'], regression_artifacts / 'XGB') 

    regression_model_dict={'LSTM': lstm_regression_result, 'CNN': cnn_regression_result, 'XGBoost': xgb_regression_result}

    # model comparison
    regression_comparison_df=mc_r.model_comparison(regression_model_dict, regression_artifacts, regression_config['model_comparison'])
    mc_r.locate_best_model(regression_comparison_df, regression_model_dict, regression_time_dimension_dict, regression_config['model_comparison'], regression_artifacts)

