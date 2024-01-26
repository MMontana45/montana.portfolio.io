'''
Title: DSC 450 Project
Author: Michael J. Montana
Date: 02 Mar 2024
Modified By: N/A
Description: Creating a Predictive Model Dash Application
'''
import pandas as pd
import dash
from dash import dcc, html
# import dash_core_components as dcc
# import dash_html_components as html
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc
import joblib
from sklearn.preprocessing import StandardScaler

# Load the scaler and model
scaler = joblib.load('../model/scaler.pk1')
rf_model = joblib.load('../model/rf_best_model.pk1')

# Initialize the app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.CYBORG])  # Dark theme

# Define the app layout
app.layout = dbc.Container(children=[
    # Row 1: Title Header
    dbc.Row([dbc.Col(html.H1('Insurance Premium Cost Prediction', style={'textAlign': 'center'}))]),

    # Row 2: Input Features
    dbc.Row([
        dbc.Col([
            # Input 1: Age
            dbc.Row([
                dbc.Col(html.Label('Age:', style={'fontWeight': 'bold','textAlign':'right'}), width=2),
                dbc.Col(dcc.Input(id='age', type='number', value=''), width=10)
            ] ,style={'marginBottom': '10px'}),

            # Input 2: Sex
            dbc.Row([
                dbc.Col(html.Label('Sex:', style={'fontWeight': 'bold','textAlign':'right'}), width=2),
                dbc.Col(dcc.RadioItems(
                    id='sex',
                    options=[
                        {'label': 'Female', 'value': 0},
                        {'label': 'Male', 'value': 1}
                    ],
                    value=0  # Default value (Female)
                ), width=10)
            ],style={'marginBottom': '10px'}),

            # Input 3: BMI
            dbc.Row([
                dbc.Col(html.Label('BMI:', style={'fontWeight': 'bold','textAlign':'right'}), width=2),
                dbc.Col(dcc.Input(id='bmi', type='number', value=''), width=10)
            ],style={'marginBottom': '10px'}),

            # Input 4: Number of Children
            dbc.Row([
                dbc.Col(html.Label('Number of Children:', style={'fontWeight': 'bold','textAlign':'right'}), width=2),
                dbc.Col(dcc.Input(id='children', type='number', value=''), width=10)
            ],style={'marginBottom': '10px'}),

            # Input 5: Smoker
            dbc.Row([
                dbc.Col(html.Label('Smoker:', style={'fontWeight': 'bold','textAlign':'right'}), width=2),
                dbc.Col(dcc.RadioItems(
                    id='smoker',
                    options=[
                        {'label': 'No', 'value': 0},
                        {'label': 'Yes', 'value': 1}
                    ],
                    value=0  # Default value (No)
                ), width=10)
            ],style={'marginBottom': '10px'}),

            # Input 6: Region
            dbc.Row([
                dbc.Col(html.Label('Region:', style={'fontWeight': 'bold','textAlign':'right'}), width=2),
                dbc.Col(dcc.RadioItems(
                    id='region',
                    options=[
                        {'label': 'Northeast', 'value': 'northeast'},
                        {'label': 'Northwest', 'value': 'northwest'},
                        {'label': 'Southeast', 'value': 'southeast'},
                        {'label': 'Southwest', 'value': 'southwest'}
                    ],
                    value='northwest'  # Default value
                ), width=10)
            ],style={'marginBottom': '10px'}),

            # Submit Button
            dbc.Row([
                dbc.Col(width=2),
                dbc.Col(html.Button('Predict Premium', id='predict-button', n_clicks=0), width=10)
            ])
        ], width=12)
    ]),

    # Row 3: Output Premium
    dbc.Row([
        dbc.Col(html.Div(id='premium-output', style={'margin-top': '10px'}), width=12)
    ])
])

# Callback function for prediction
@app.callback(
    Output('premium-output', 'children'),
    [Input('predict-button', 'n_clicks')],
    [dash.Input('age', 'value'),
     dash.Input('sex', 'value'),
     dash.Input('bmi', 'value'),
     dash.Input('children', 'value'),
     dash.Input('smoker', 'value'),
     dash.Input('region', 'value')])

def prediction(n_clicks, age, sex, bmi, children, smoker, region):
    if n_clicks > 0:
        # Convert region to binary values
        region_values = {
            'northeast': [1, 0, 0, 0],
            'northwest': [0, 1, 0, 0],
            'southeast': [0, 0, 1, 0],
            'southwest': [0, 0, 0, 1]}
        region_binary = region_values[region]

        # Ensure this matches the structure of your training data
        input_data = pd.DataFrame({
            'age': [age],
            'sex': [sex],
            'bmi': [bmi],
            'children': [children],
            'smoker': [smoker],
            'region_northeast': [region_binary[0]],
            'region_northwest': [region_binary[1]],
            'region_southeast': [region_binary[2]],
            'region_southwest': [region_binary[3]]})

        # Scale and predict
        input_data_scaled = scaler.transform(input_data)
        premium_prediction = rf_model.predict(input_data_scaled)[0]
        # premium_prediction = rf_model.predict(input_data)
        print(f"Predicted Premium: ${premium_prediction:.2f}")
        return f'Predicted Premium: ${premium_prediction:.2f}'
    else:
        print("No prediction made yet")
        return ''

if __name__ == '__main__':
    app.run_server(debug=True)
