import numpy as np
import argparse
import streamlit as st
import matplotlib.pyplot as plt
import os

import yaml

from classification.inference import load_data as ld, \
    data_cleaning as dc, time_dimension as td, \
    load_artifacts as la, process_input as pi
from regression.inference import load_artifacts as la_r, process_input as pi_r

def predict_failure(input, model, original_data, config):
    if hasattr(model, 'predict_proba'):
        # Model has predict_proba() function
        prediction_prob = model.predict_proba(input)
    else:
        # Model does not have predict_proba() function, use predict() instead
        prediction_prob = model.predict(input)
    original_data['Probability(%)']=np.round(100*prediction_prob,2)
    original_data['Predicted_Failure']=(prediction_prob >= config['decision_boundary']).astype(int)
    return original_data

def predict_RUL(input, model, original_data):
    """create prediction based on input and model""" 
    prediction = model.predict(input)
    original_data['Predicted_RUL (cycles)'] = np.round(prediction.flatten(),4)
    return original_data

def predict(input, model):
    """create prediction based on input and model""" 
    prediction = model.predict(input)
    return prediction

def make_grid(cols,rows):
    """create desired grid for display""" 
    grid = [0]*cols
    for i in range(cols):
        with st.container():
            grid[i] = st.columns(rows)
    return grid

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Acquire, clean, and create features from RUL data"
    )
    parser.add_argument(
        "--config", default="config/inference-config.yaml", help="Path to configuration file"
    )
    args = parser.parse_args()

    # Load configuration file for parameters and run config
    with open(args.config, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    
    if "selected_option" not in st.session_state:
        st.session_state["selected_option"] = None

    # create two options
    st.sidebar.subheader('Please Select Prediction Based on Local Data Path or Manual Input')
    if st.sidebar.button('Local Data Path'):
        st.session_state["selected_option"] = "entire_data"
    if st.sidebar.button('Manual Input: Failure'):
        st.session_state["selected_option"] = "manual_input_failure"
    if st.sidebar.button('Manual Input: RUL'):
        st.session_state["selected_option"] = "manual_input_rul"

    if st.session_state["selected_option"] == "entire_data":
        # First Section: display predicted df
        with st.expander("Failure", expanded=True):

            st.subheader('Section 1: Failure Prediction')
            st.write('\n')

            # load in datasets
            data_path = st.sidebar.text_input("Please Enter Path To Your Data")
            if st.sidebar.button('Enter'):
                df=ld.load_data(f'{data_path}', config['load_data'])
                # sample: data/ALLtestMescla5D.csv

                # clean data
                data_cleaning_config = config.get("data_cleaning")
                if data_cleaning_config.get("detect_negative", False):
                    df=dc.drop_negative(df, config['data_cleaning'])
                df_cleaned=dc.data_cleaning(df)

                # create different time dimension df
                st.session_state["time_dimension_dict"] =td.create_time_dimension(df_cleaned)
            
            if "time_dimension_dict" in st.session_state:
                # define first user selection: time
                st.sidebar.subheader("Time Granularity")
                time_version = os.getenv("DEFAULT_TIME_GRANULARITY", config['app']['default_time'])
                # Find available model versions in artifacts dir
                available_times = list(st.session_state.get("time_dimension_dict", {}).keys())
                # Create a dropdown to select the model
                time_version = st.sidebar.selectbox("Select Time Granularity", list(available_times))
                st.write(f"Selected time granularity: {time_version}")

                # define second user selection: model
                st.sidebar.subheader("Failure Prediction Model")
                model_version = os.getenv("DEFAULT_MODEL_VERSION", config['app']['failure_default_model'])
                # Find available model versions in artifacts dir
                available_models = config['app']['failure_available_models']
                # Create a dropdown to select the model
                model_version = st.sidebar.selectbox("Select Model", list(available_models))
                st.write(f"Selected model version: {model_version}")

                st.session_state['time_version'] = time_version
                st.session_state['model_version'] = model_version

                if st.sidebar.button('Predict Failure'):
                    # load in data
                    data=st.session_state["time_dimension_dict"][time_version]
                    data_for_prediction=data.drop(columns=['machineID','datetime'])

                    # load in artifacts
                    model, scaler, encoder, columns = la.load_artifacts(time_version, model_version, "classification_artifacts", config['app'])

                    data_for_prediction=data_for_prediction[columns]
                    processed_input = pi.process_input(data_for_prediction, scaler, encoder, config['app']['one_hot_encode'])
                    # make prediction
                    prediction = predict_failure(processed_input, model, data, config['app'])

                    # reorder columns
                    first_columns_order = config['app']['classification_first_columns_order']
                    remaining_columns = [i for i in prediction.columns if i not in first_columns_order]
                    desired_order = first_columns_order + remaining_columns
                    prediction = prediction.reindex(columns=desired_order)
                    # for_regression.to_csv(f'data/{time_version}_{model_version}_prediction.csv', index=False)

                    # select max datetime for each machineID
                    maxdate_index=prediction.groupby('machineID')['datetime'].idxmax()
                    prediction_maxdate=prediction.loc[maxdate_index]
                    prediction_maxdate=prediction_maxdate.drop(columns=['datetime']).reset_index(drop=True)
                    # sort by prediction in ascending order
                    prediction_maxdate.sort_values(by=['Predicted_Failure','Probability(%)'], ascending=[False, False], inplace=True, ignore_index=True)

                    # save prediction data at this point for 1) directly pass in to regression 2) with datetime
                    failed_machines=prediction_maxdate[prediction_maxdate['Predicted_Failure']==1]['machineID'].unique()
                
                    # Store prediction in session state
                    st.session_state["prediction"] = prediction
                    st.session_state["prediction_maxdate"] = prediction_maxdate
                    st.session_state["failed_machines"] = failed_machines

                    # highlight prediction<7 in red and else in green
                    def color_predictions(val):
                        color = 'red' if val==1 else 'green'
                        return f'color: {color}'
                    
                    if len(failed_machines)!=0:
                        st.write(f'These machines are predicted to fail:{failed_machines}')
                    else:
                        st.write('All machines are safe')

                    st.table(prediction_maxdate.style.applymap(color_predictions, subset=['Predicted_Failure']))
                    # st.write(f"Prediction csv file saved to path: data/{time_version}_{model_version}_prediction.csv")
        
        with st.expander("RUL", expanded=True):
 
            st.subheader('Section 2: RUL Predictionl')
            st.write('\n')

            if "failed_machines" in st.session_state:
                # define third user selection: model
                st.sidebar.subheader("RUL Prediction Model")
                rul_model_version = os.getenv("DEFAULT_MODEL_VERSION", config['app']['rul_default_model'])
                # Find available model versions in artifacts dir
                rul_available_models = config['app']['rul_available_models']
                # Create a dropdown to select the model
                rul_model_version = st.sidebar.selectbox("Select Model", list(rul_available_models))
                st.write(f"Selected model version: {rul_model_version}")

                time_version=st.session_state['time_version']
                # st.session_state['model_version'] = rul_model_version

                if st.sidebar.button('Predict RUL'):
                    # load in data

                    data=st.session_state["prediction"]
                    data_for_prediction=data.drop(columns=['machineID','datetime','Predicted_Failure', 'Probability(%)'])
                    
                    # load in artifacts
                    model, scaler, encoder, columns = la_r.load_artifacts(time_version, rul_model_version, "regression_artifacts", config['app'])

                    data_for_prediction=data_for_prediction[columns]
                    processed_input = pi_r.process_input(data_for_prediction, scaler, encoder, config['app']['one_hot_encode'])
                    # make prediction
                    prediction = predict_RUL(processed_input, model, data)

                    # reorder columns
                    first_three_columns_order = ['machineID', 'Predicted_RUL (cycles)','RUL']
                    remaining_columns = [i for i in prediction.columns if i not in first_three_columns_order]
                    desired_order = first_three_columns_order + remaining_columns
                    prediction = prediction.reindex(columns=desired_order)
                    prediction = prediction.rename(columns={'RUL':'Cycles Since Last Failure'})

                    # select max datetime for each machineID
                    maxdate_index=prediction.groupby('machineID')['datetime'].idxmax()
                    prediction_maxdate=prediction.loc[maxdate_index]
                    prediction_maxdate=prediction_maxdate.drop(columns=['datetime','Predicted_Failure', 'Probability(%)']).reset_index(drop=True)
                    # only display not failed machine RUL values
                    failed_machines=st.session_state["failed_machines"]
                    prediction_maxdate=prediction_maxdate[~prediction_maxdate['machineID'].isin(failed_machines)]
                    # sort by prediction in ascending order
                    prediction_maxdate.sort_values(by='Predicted_RUL (cycles)', ascending=True, inplace=True, ignore_index=True)

                    # Store prediction in session state
                    st.session_state["full_prediction"] = prediction

                    # highlight prediction<cutoff in red and else in green
                    def color_predictions(val):
                        color = 'red' if val < config['app']['cutoff_value'] else 'green'
                        return f'color: {color}'
                    
                    endangered_machines=prediction_maxdate[prediction_maxdate['Predicted_RUL (cycles)']<config['app']['cutoff_value']]['machineID'].unique()
                    st.session_state["endangered_machines"] = endangered_machines

                    if len(endangered_machines)!=0:
                        st.write(f'These machines are endangered:{endangered_machines}')
                    else:
                        st.write('All machines are safe')

                    st.table(prediction_maxdate.style.applymap(color_predictions, subset=['Predicted_RUL (cycles)']))

        # Third Section: plot
        with st.expander("Plot", expanded=True):
            if "full_prediction" in st.session_state:
                st.subheader('Section 3: Plots')
                st.write('\n')

                full_prediction = st.session_state["full_prediction"]
                failed=st.session_state["failed_machines"]
                endangered = st.session_state["endangered_machines"]
                all_machine = full_prediction['machineID'].unique()
                machine_choice={'Failed':failed, 'Endangered':endangered, 'All_Machine': all_machine}

                machine_range = st.selectbox("Select Failed / Endangered / All Machines", list(machine_choice.keys()))
                selected_machine = st.selectbox("Select Machine", machine_choice[machine_range])

                # define plotting detail
                unplottable=['model','age','machineID']
                if machine_range=='Failed':
                    plot_features = [i for i in full_prediction.columns if i not in unplottable+['Predicted_RUL (cycles)']]
                    # Select x-axis feature
                    selected_x_feature = 'datetime'
                    y_options = [feature for feature in plot_features if feature != selected_x_feature]
                    # Select y-axis features
                    selected_y_features = st.multiselect("Select Y-axis Features", y_options, default=['Predicted_Failure'])
                else:
                    plot_features = [i for i in full_prediction.columns if i not in unplottable]
                    # Select x-axis feature
                    selected_x_feature = 'datetime'
                    y_options = [feature for feature in plot_features if feature != selected_x_feature]
                    # Select y-axis features
                    selected_y_features = st.multiselect("Select Y-axis Features", y_options, default=['Predicted_RUL (cycles)'])

                # Add a plot button
                plot_button = st.button("Plot")

                if plot_button:
                    # Filter data for the selected machine and features
                    filtered_data = full_prediction[full_prediction['machineID'] == selected_machine][[selected_x_feature] + selected_y_features]
                    filtered_data.sort_values(by='datetime', ascending=True, ignore_index=True, inplace=True)

                    # Create a new matplotlib figure and axis
                    fig = plt.figure()

                    # For each selected y feature, plot a line on the axis
                    for feature in selected_y_features:
                        plt.plot(selected_x_feature, feature, data=filtered_data, label=feature)

                    # Set the x-axis label to the selected x feature
                    plt.xlabel(selected_x_feature)
                    plt.xticks(rotation = 45)

                    # Create a legend
                    plt.legend()

                    # Display the figure in Streamlit
                    st.pyplot(fig)

    if st.session_state["selected_option"] == "manual_input_failure":
        # Third Section: manual input
        # with st.expander("Manual Value Prediction", expanded=True):
        st.subheader('Failure Prediction Based on Entered Values')
        st.write('\n')

        # define first user selection: time
        st.sidebar.subheader("Time Granularity Selection")
        time_version = os.getenv("DEFAULT_TIME_GRANULARITY", config['app']['default_time'])
        # Find available model versions in artifacts dir
        available_times = config['app']['available_times']
        # Create a dropdown to select the model
        time_version = st.sidebar.selectbox("Select Time Granularity", list(available_times))
        st.write(f"Selected time granularity: {time_version}")

        # define second user selection: model
        st.sidebar.subheader("Model Selection")
        model_version = os.getenv("DEFAULT_MODEL_VERSION", config['app']['failure_default_model'])
        # Find available model versions in artifacts dir
        available_models = config['app']['failure_available_models']
        # Create a dropdown to select the model
        model_version = st.sidebar.selectbox("Select Model", list(available_models))
        st.write(f"Selected model version: {model_version}")

        st.write('Please enter the following values for the machine that you want to predict')

        # load the model and preprocessors
        model, scaler, encoder, columns = la.load_artifacts(time_version, model_version, "classification_artifacts", config['app'])

        # set up user_input dict
        user_input={}

        # populate grid
        display_grid = make_grid(3,4)
        # First row
        columns_first_row = columns[:3]
        for i in range(len(columns_first_row)):
            input_value = display_grid[0][i].text_input(f"Enter value for {columns_first_row[i]}")
            user_input[f'{columns_first_row[i]}'] = input_value
        # Second row
        columns_second_row = columns[3:7]
        for i in range(len(columns_second_row)):
            input_value = display_grid[1][i].text_input(f"Enter value for {columns_second_row[i]}")
            user_input[f'{columns_second_row[i]}'] = input_value
        # Third row
        columns_third_row = columns[7:]
        for i in range(len(columns_third_row)):
            input_value = display_grid[2][i].text_input(f"Enter value for {columns_third_row[i]}")
            user_input[f'{columns_third_row[i]}'] = input_value
        st.markdown('---')

        if st.button('Predict Failure'):
            processed_input = pi.process_input(user_input, scaler, encoder, config['app']['one_hot_encode'])
            if hasattr(model, 'predict_proba'):
                # Model has predict_proba() function
                prediction_prob = model.predict_proba(processed_input)
            else:
                # Model does not have predict_proba() function, use predict() instead
                prediction_prob = model.predict(processed_input)
            prediction_prob = float(prediction_prob[0])
            prediction=int(prediction_prob >= config['app']['decision_boundary'])
            color = 'green' if prediction ==0 else 'red'
            if prediction==0:
                st.markdown(f'<p style="color:{color}; font-size:20px;">The Machine is NOT GOING TO FAIL</p>', unsafe_allow_html=True)
            else:
                st.markdown(f'<p style="color:{color}; font-size:20px;">The Machine is PREDICTED TO FAIL, with failure probability {100 * prediction_prob:.2f}%</p>', unsafe_allow_html=True)
        
    if st.session_state["selected_option"] == "manual_input_rul":
        # Third Section: manual input
        # with st.expander("Manual Value Prediction", expanded=True):
        st.subheader('RUL Prediction Based on Entered Values')
        st.write('\n')

        # define first user selection: time
        st.sidebar.subheader("Time Granularity Selection")
        time_version = os.getenv("DEFAULT_TIME_GRANULARITY", config['app']['default_time'])
        # Find available model versions in artifacts dir
        available_times = config['app']['available_times']
        # Create a dropdown to select the model
        time_version = st.sidebar.selectbox("Select Time Granularity", list(available_times))
        st.write(f"Selected time granularity: {time_version}")

        # define second user selection: model
        st.sidebar.subheader("Model Selection")
        model_version = os.getenv("DEFAULT_MODEL_VERSION", config['app']['rul_default_model'])
        # Find available model versions in artifacts dir
        available_models = config['app']['rul_available_models']
        # Create a dropdown to select the model
        model_version = st.sidebar.selectbox("Select Model", list(available_models))
        st.write(f"Selected model version: {model_version}")

        st.write('Please enter the following values for the machine that you want to predict')

        # load the model and preprocessors
        model, scaler, encoder, columns = la_r.load_artifacts(time_version, model_version, "regression_artifacts", config['app'])

        # set up user_input dict
        user_input={}

        # populate grid
        display_grid = make_grid(3,4)
        # First row
        columns_first_row = columns[:3]
        for i in range(len(columns_first_row)):
            if columns_first_row[i]=='RUL':
                input_value = display_grid[0][i].text_input(f"Enter value for Cycles Since Last Failure")
            else:
                input_value = display_grid[0][i].text_input(f"Enter value for {columns_first_row[i]}")
            user_input[f'{columns_first_row[i]}'] = input_value
        # Second row
        columns_second_row = columns[3:7]
        for i in range(len(columns_second_row)):
            input_value = display_grid[1][i].text_input(f"Enter value for {columns_second_row[i]}")
            user_input[f'{columns_second_row[i]}'] = input_value
        # Third row
        columns_third_row = columns[7:]
        for i in range(len(columns_third_row)):
            input_value = display_grid[2][i].text_input(f"Enter value for {columns_third_row[i]}")
            user_input[f'{columns_third_row[i]}'] = input_value
        st.markdown('---')

        if st.button('Predict RUL'):
            processed_input = pi_r.process_input(user_input, scaler, encoder, config['app']['one_hot_encode'])
            prediction = predict(processed_input, model)
            prediction_value = float(prediction[0])
            color = 'green' if prediction_value >= config['app']['cutoff_value'] else 'red'
            st.markdown(f'<p style="color:{color}; font-size:20px;">The predicted RUL based on your entered values is: \
                        {prediction_value:.2f} cycles</p>', unsafe_allow_html=True)