# Remaining Useful Life (RUL) Prediction

## Overview
This repo aims at providing an initial structure of training and inference pipelines based on manufacturing data with time-series nature. The product of this repo is a streamlit web-app that allows users to pass in 1) data file or 2) manual entries of data values for prediction of Failure and RUL. 

Some key design features of the pipelines are summarized below:
1. Configuration: pipeline parameters
All configurations and parameters used during model development are stored in **regression-config.yaml** and **classification-config.yaml**. All configurations and parameters used during model inference are stored in **inference-config.yaml**. You could put additional parameters into the file (following the format and restrictions currently set up in the file) without changing the src code. 

2. Configuration: logging
Logger object information are set up through **local.conf**. All current logging messages are printed through the terminal. You may change the level, handler, or add additional logging files for your own purposes.

3. Time Dimension
Different time dimension based on the time-series nature of the data is of key importance to making trust-worthy predictions. Two sets of time dimensions (24h / 5d) are made possible based on the provided data. 

4. Models
Three different models are being evaluated in both Failure classification and RUL regression. Feel free to add additional models, as long as another model result dict is present in the **train_pipeline.py** following similar format. 

### Environmental set-up
The only necessary step is to go through the **requirements.txt** file and install all necessary packages. 

### Run the training pipeline
To run the training pipeline, simply enter: 
```bash
python3 train_pipeline.py
```

### Run the inference pipeline
To run the inference pipeline, simply enter:
```bash
streamlit run app.py
```
