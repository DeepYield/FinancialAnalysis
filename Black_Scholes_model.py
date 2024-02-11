from dash import Dash, dcc, html, Input, Output, State, callback
import numpy as np
import pandas as pd
from scipy.stats import norm

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

def generate_table(dataframe, max_rows=10):
    return html.Table([
        html.Thead(
            html.Tr([html.Th(col) for col in dataframe.columns])
        ),
        html.Tbody([
            html.Tr([
                html.Td(dataframe.iloc[i][col]) for col in dataframe.columns
            ]) for i in range(min(len(dataframe), max_rows))
        ])
    ])


app = Dash(__name__, external_stylesheets=external_stylesheets)
server = app.server

app.layout = html.Div([
    html.H1('Black-Scholes Option Pricing Model', style={'textAlign': 'center', 'color': '#11111'}),
    dcc.Input(id='input-1-state', type='text', value='Price'),
    dcc.Input(id='input-2-state', type='text', value='Strike'),
    dcc.Input(id='input-3-state', type='text', value='Vol (std)'),
    dcc.Input(id='input-4-state', type='text', value='Interest Rate (bps)'),
    dcc.Input(id='input-5-state', type='text', value='Time (yrs)'),
    dcc.Input(id='input-6-state', type='text', value='Dividend Yield (bps)'),
    html.Button(id='submit-button-state', n_clicks=0, children='Submit'),
    html.Div(id='output-state')
],
    style={
        'display': 'flex',
        'flex-direction': 'column',
        'align-items' : 'flex-start',
        'margin': '20px'  # Optional: add margin around the container
    }
)


@callback(Output('output-state', 'children'),
              Input('submit-button-state','n_clicks'),
              State('input-1-state', 'value'),
              State('input-2-state', 'value'),
              State('input-3-state', 'value'),
              State('input-4-state', 'value'),
              State('input-5-state', 'value'),
              State('input-6-state', 'value'))

def update_variables(click,input1,input2,input3,input4,input5,input6):
    if input1 == 'Price':
        pass
    else:
        S = float(input1)
        K = float(input2)
        vol = float(input3)
        rfr = float(input4)
        t = float(input5)
        q = float(input6)
        
        #Call and put formula terms
        nom_1 = np.log(S/K)
        nom_2 = t*(rfr - q + ((vol**2)/2))
        denom = (vol*np.sqrt(t))
        d1 = (nom_1 + nom_2) / denom
        d2 = d1 - denom
        c_term_1 = (S*np.exp(-q*t)) * norm.cdf(d1)
        c_term_2 = (K*np.exp(-rfr*t)) * norm.cdf(d2)
        p_term_2 = (S*np.exp(-q*t)) * norm.cdf(-d1)
        p_term_1 = (K*np.exp(-rfr*t)) * norm.cdf(-d2)
        call = round(c_term_1 - c_term_2,2)
        put = round(p_term_1 - p_term_2,2)
        
        #Call and Put Delta
        c_delta = round(np.exp(-q*t)*norm.cdf(d1),2)
        p_delta = round(np.exp(-q*t)*(norm.cdf(d1)-1),2)
        
        #Call and Put Gamma is the same for both
        g_term1 = norm.pdf(-d1)
        g_term2 = (np.exp(-q*t))/(S*denom)
        gamma = round(g_term2*g_term1,2)
        
        #Call and Put Theta
        c_theta = round((-(norm.pdf(-d1)*(S*vol*np.exp(-q*t)/(2*np.sqrt(t))))-rfr*c_term_2+q*c_term_1),2)
        p_theta = round((-(norm.pdf(-d1)*(S*vol*np.exp(-q*t)/(2*np.sqrt(t))))+rfr*p_term_1-q*p_term_2),2)
        
        #Call and Put Vega per 1% point change in volatility 
        vega = round(0.01*S*np.exp(-q*t)*np.sqrt(t)*norm.pdf(d1),2)
        
        #Call and Put Rho per 1% point change in interest rates
        c_rho = round(0.01*c_term_2*t,2)
        p_rho = round(0.01*p_term_1*t,2)
        
        #Dataframe for table output
        output_dict = {
        'Call Price': [call],
        'Call Delta': [c_delta],
        'Call Theta': [c_theta],
        'Call Rho': [c_rho],
        'Put Price': [put],
        'Put Delta': [p_delta],
        'Put Theta': [p_theta],
        'Put Rho': [p_rho],
        'Gamma': [gamma],
        'Vega': [vega]
        }

        df = pd.DataFrame(output_dict)

        # Return the calculated variables
        return generate_table(df)


if __name__ == '__main__':
    app.run(debug=True)
